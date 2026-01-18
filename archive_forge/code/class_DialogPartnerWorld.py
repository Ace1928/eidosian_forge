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
class DialogPartnerWorld(World):
    """
    Simple world for two agents communicating synchronously.

    This basic world switches back and forth between two agents, giving each agent one
    chance to speak per turn and passing that back to the other one.
    """

    def __init__(self, opt: Opt, agents=None, shared=None):
        if not (agents is not None) ^ (shared is not None):
            raise ValueError('You must supply either agents or shared, but not both.')
        super().__init__(opt)
        if shared:
            self.agents = create_agents_from_shared(shared['agents'])
        else:
            if len(agents) != 2:
                raise RuntimeError('There must be exactly two agents for this world.')
            self.agents = agents
        self.acts = [None] * len(self.agents)
        if self.agents is not None and len(self.agents) > 0:
            self.id = self.get_task_agent().getID()

    def get_task_agent(self):
        """
        Return task agent.
        """
        return self.get_agents()[0]

    def get_model_agent(self):
        """
        Return model agent, if applicable.
        """
        return self.get_agents()[1]

    def parley(self):
        """
        Agent 0 goes first.

        Alternate between the two agents.
        """
        acts = self.acts
        agents = self.agents
        acts[0] = agents[0].act()
        agents[1].observe(validate(acts[0]))
        acts[1] = agents[1].act()
        agents[0].observe(validate(acts[1]))
        self.update_counters()

    def episode_done(self):
        """
        Only the first agent indicates when the episode is done.
        """
        if self.acts[0] is not None:
            return self.acts[0].get('episode_done', False)
        else:
            return False

    def epoch_done(self):
        """
        Only the first agent indicates when the epoch is done.
        """
        return self.get_task_agent().epoch_done()

    def report(self):
        """
        Report all metrics of all subagents.
        """
        from parlai.core.metrics import Metric, LegacyMetric
        metrics = {}
        for a in self.agents:
            if hasattr(a, 'report'):
                m = a.report()
                for k, v in m.items():
                    if not isinstance(v, Metric):
                        v = LegacyMetric(v)
                    if k not in metrics:
                        metrics[k] = v
        if metrics and 'exs' in metrics:
            self.total_exs += metrics['exs'].value()
        return metrics

    def num_examples(self):
        """
        Return number of examples.
        """
        if hasattr(self.get_task_agent(), 'num_examples'):
            return self.get_task_agent().num_examples()
        return 0

    def num_episodes(self):
        """
        Return number of episodes.
        """
        if hasattr(self.get_task_agent(), 'num_episodes'):
            return self.get_task_agent().num_episodes()
        return 0

    def shutdown(self):
        """
        Shutdown each agent.
        """
        for a in self.agents:
            a.shutdown()

    def update_counters(self):
        """
        Ensure all counters are synchronized across threads.
        """
        super().update_counters()
        for a in self.agents:
            if hasattr(a, 'update_counters'):
                a.update_counters()