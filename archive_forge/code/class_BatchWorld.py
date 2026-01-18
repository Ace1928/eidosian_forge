import copy
import random
from typing import List, Dict, Union
from parlai.core.agents import create_agents_from_shared
from parlai.core.loader import load_task_module, load_world_module
from parlai.core.metrics import aggregate_named_reports
from parlai.core.opt import Opt
from parlai.core.teachers import Teacher, create_task_agent_from_taskname
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import Timer, display_messages
from parlai.tasks.tasks import ids_to_tasks
import parlai.utils.logging as logging
class BatchWorld(World):
    """
    BatchWorld contains many copies of the same world.

    Create a separate world for each item in the batch, sharing
    the parameters for each.

    The underlying world(s) it is batching can be either
    ``DialogPartnerWorld``, ``MultiAgentWorld``, or ``MultiWorld``.
    """

    def __init__(self, opt: Opt, world):
        super().__init__(opt)
        self.opt = opt
        self.random = opt.get('datatype', None) == 'train'
        self.world = world
        self.worlds: List[World] = []
        for i in range(opt['batchsize']):
            shared = world.share()
            shared['batchindex'] = i
            for agent_shared in shared.get('agents', ''):
                agent_shared['batchindex'] = i
            _override_opts_in_shared(shared, {'batchindex': i})
            self.worlds.append(shared['world_class'](opt, None, shared))
        self.batch_observations = [None] * len(self.world.get_agents())
        self.first_batch = None
        self.acts = [None] * len(self.world.get_agents())

    def batch_observe(self, index, batch_actions, index_acting):
        """
        Observe corresponding actions in all subworlds.
        """
        batch_observations = []
        for i, w in enumerate(self.worlds):
            agents = w.get_agents()
            observation = None
            if batch_actions[i] is None:
                batch_actions[i] = [{}] * len(self.worlds)
            if hasattr(w, 'observe'):
                observation = w.observe(agents[index], validate(batch_actions[i]))
            else:
                observation = validate(batch_actions[i])
            if index == index_acting:
                if hasattr(agents[index], 'self_observe'):
                    agents[index].self_observe(observation)
            else:
                observation = agents[index].observe(observation)
            if observation is None:
                raise ValueError('Agents should return what they observed.')
            batch_observations.append(observation)
        return batch_observations

    def batch_act(self, agent_idx, batch_observation):
        """
        Act in all subworlds.
        """
        a = self.world.get_agents()[agent_idx]
        if hasattr(a, 'batch_act'):
            batch_actions = a.batch_act(batch_observation)
            for i, w in enumerate(self.worlds):
                acts = w.get_acts()
                acts[agent_idx] = batch_actions[i]
        else:
            batch_actions = []
            for w in self.worlds:
                agents = w.get_agents()
                acts = w.get_acts()
                acts[agent_idx] = agents[agent_idx].act()
                batch_actions.append(acts[agent_idx])
        return batch_actions

    def parley(self):
        """
        Parley in all subworlds.

        Usually with ref:`batch_act` and ref:`batch_observe`.
        """
        num_agents = len(self.world.get_agents())
        batch_observations = self.batch_observations
        if hasattr(self.world, 'parley_init'):
            for w in self.worlds:
                w.parley_init()
        for agent_idx in range(num_agents):
            batch_act = self.batch_act(agent_idx, batch_observations[agent_idx])
            self.acts[agent_idx] = batch_act
            if hasattr(self.world, 'execute'):
                for w in self.worlds:
                    w.execute(w.agents[agent_idx], batch_act[agent_idx])
            for other_index in range(num_agents):
                obs = self.batch_observe(other_index, batch_act, agent_idx)
                if obs is not None:
                    batch_observations[other_index] = obs
        self.update_counters()

    def display(self):
        """
        Display the full batch.
        """
        s = '[--batchsize ' + str(len(self.worlds)) + '--]\n'
        for i, w in enumerate(self.worlds):
            s += '[batch world ' + str(i) + ':]\n'
            s += w.display() + '\n'
        s += '[--end of batch--]'
        return s

    def num_examples(self):
        """
        Return the number of examples for the root world.
        """
        return self.world.num_examples()

    def num_episodes(self):
        """
        Return the number of episodes for the root world.
        """
        return self.world.num_episodes()

    def get_total_exs(self):
        """
        Return the total number of processed episodes in the root world.
        """
        return self.world.get_total_exs()

    def getID(self):
        """
        Return the ID of the root world.
        """
        return self.world.getID()

    def get_agents(self):
        """
        Return the agents of the root world.
        """
        return self.world.get_agents()

    def get_task_agent(self):
        """
        Return task agent of the root world.
        """
        return self.world.get_task_agent()

    def get_model_agent(self):
        """
        Return model agent of the root world.
        """
        return self.world.get_model_agent()

    def episode_done(self):
        """
        Return whether the episode is done.

        A batch world is never finished, so this always returns `False`.
        """
        return False

    def epoch_done(self):
        """
        Return if the epoch is done in the root world.
        """
        if self.world.epoch_done():
            return True
        for world in self.worlds:
            if not world.epoch_done():
                return False
        return True

    def report(self):
        """
        Report metrics for the root world.
        """
        return self.world.report()

    def reset(self):
        """
        Reset the root world, and all copies.
        """
        self.world.reset()
        for w in self.worlds:
            w.reset()

    def reset_metrics(self):
        """
        Reset metrics in the root world.
        """
        self.world.reset_metrics()

    def shutdown(self):
        """
        Shutdown each world.
        """
        for w in self.worlds:
            w.shutdown()
        self.world.shutdown()