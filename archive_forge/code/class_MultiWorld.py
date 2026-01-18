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
class MultiWorld(World):
    """
    Container for multiple worlds.

    Container for a set of worlds where each world gets a turn in a round-robin fashion.
    The same user_agents are placed in each, though each world may contain additional
    agents according to the task that world represents.
    """

    def __init__(self, opt: Opt, agents=None, shared=None, default_world=None):
        super().__init__(opt)
        self.worlds: List[World] = []
        for index, k in enumerate(opt['task'].split(',')):
            k = k.strip()
            if k:
                opt_singletask = copy.deepcopy(opt)
                opt_singletask['task'] = k
                if shared:
                    s = shared['worlds'][index]
                    self.worlds.append(s['world_class'](s['opt'], None, s))
                else:
                    self.worlds.append(create_task_world(opt_singletask, agents, default_world=default_world))
        self.world_idx = -1
        self.new_world = True
        self.parleys = -1
        self.is_training = DatatypeHelper.is_training(opt.get('datatype'))
        self.cum_task_weights = [1] * len(self.worlds)
        self.task_choices = range(len(self.worlds))
        weights = self.opt.get('multitask_weights', [1])
        if weights == 'stochastic':
            weights = [w.num_episodes() for w in self.worlds]
        sum = 0
        for i in self.task_choices:
            if len(weights) > i:
                weight = weights[i]
            else:
                weight = 1
            self.cum_task_weights[i] = weight + sum
            sum += weight
        task_ids: Dict[str, Teacher] = {}
        for each_world in self.worlds:
            world_id = each_world.getID()
            if world_id in task_ids:
                raise AssertionError('{} and {} teachers have overlap in id {}.'.format(task_ids[world_id], each_world.get_agents()[0].__class__, world_id))
            else:
                task_ids[world_id] = each_world.get_task_agent()

    def num_examples(self):
        """
        Return sum of each subworld's number of examples.
        """
        if not hasattr(self, 'num_exs'):
            worlds_num_exs = [w.num_examples() for w in self.worlds]
            if any((num is None for num in worlds_num_exs)):
                self.num_exs = None
            else:
                self.num_exs = sum(worlds_num_exs)
        return self.num_exs

    def num_episodes(self):
        """
        Return sum of each subworld's number of episodes.
        """
        if not hasattr(self, 'num_eps'):
            worlds_num_eps = [w.num_episodes() for w in self.worlds]
            if any((num is None for num in worlds_num_eps)):
                self.num_eps = None
            else:
                self.num_eps = sum(worlds_num_eps)
        return self.num_eps

    def get_agents(self):
        """
        Return the agents in the *current* subworld.
        """
        return self.worlds[self.world_idx].get_agents()

    def get_task_agent(self):
        """
        Not possible/well-defined in this setting.
        """
        return self.worlds[self.world_idx].get_task_agent()

    def get_model_agent(self):
        """
        Not implemented.
        """
        return self.worlds[self.world_idx].get_model_agent()

    def get_acts(self):
        """
        Return the acts in the *current* subworld.
        """
        return self.worlds[self.world_idx].get_acts()

    def share(self):
        """
        Share all the subworlds.
        """
        shared_data = {}
        shared_data['world_class'] = type(self)
        shared_data['opt'] = self.opt
        shared_data['worlds'] = [w.share() for w in self.worlds]
        return shared_data

    def epoch_done(self):
        """
        Return if *all* the subworlds are done.
        """
        for t in self.worlds:
            if not t.epoch_done():
                return False
        return True

    def parley_init(self):
        """
        Update the current subworld.

        If we are in the middle of an episode, keep the same world and finish this
        episode. If we have finished this episode, pick a new world (either in a random
        or round-robin fashion).
        """
        self.parleys = self.parleys + 1
        if self.world_idx >= 0 and self.worlds[self.world_idx].episode_done():
            self.new_world = True
        if self.new_world:
            self.new_world = False
            self.parleys = 0
            if self.is_training:
                self.world_idx = random.choices(self.task_choices, cum_weights=self.cum_task_weights)[0]
            else:
                for _ in range(len(self.worlds)):
                    self.world_idx = (self.world_idx + 1) % len(self.worlds)
                    if not self.worlds[self.world_idx].epoch_done():
                        break

    def parley(self):
        """
        Parley the *current* subworld.
        """
        self.parley_init()
        self.worlds[self.world_idx].parley()
        self.update_counters()

    def display(self):
        """
        Display all subworlds.
        """
        if self.world_idx != -1:
            s = ''
            w = self.worlds[self.world_idx]
            if self.parleys == 0:
                s = '[world ' + str(self.world_idx) + ':' + w.getID() + ']\n'
            s = s + w.display()
            return s
        else:
            return ''

    def report(self):
        """
        Report aggregate metrics across all subworlds.
        """
        metrics = aggregate_named_reports({w.getID(): w.report() for w in self.worlds}, micro_average=self.opt.get('aggregate_micro', False))
        if 'exs' in metrics:
            self.total_exs += metrics['exs'].value()
        return metrics

    def reset(self):
        """
        Reset all subworlds.
        """
        for w in self.worlds:
            w.reset()

    def reset_metrics(self):
        """
        Reset metrics in all subworlds.
        """
        for w in self.worlds:
            w.reset_metrics()

    def update_counters(self):
        super().update_counters()
        for w in self.worlds:
            w.update_counters()