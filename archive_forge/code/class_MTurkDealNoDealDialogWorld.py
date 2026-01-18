from parlai.mturk.core.worlds import MTurkTaskWorld
from parlai.core.worlds import validate
from joblib import Parallel, delayed
from parlai.tasks.dealnodeal.agents import NegotiationTeacher
from parlai.tasks.dealnodeal.agents import get_tag
from parlai.tasks.dealnodeal.agents import WELCOME_MESSAGE
import random
class MTurkDealNoDealDialogWorld(MTurkTaskWorld):
    """
    World where two agents have a dialogue to negotiate a deal.
    """

    def __init__(self, opt, agents=None, shared=None):
        if agents is not None:
            random.shuffle(agents)
        self.agents = agents
        self.acts = [None] * len(agents)
        self.episodeDone = False
        self.task = NegotiationTeacher(opt=opt)
        self.first_turn = True
        self.choices = dict()
        self.selection = False
        self.turns = 0
        self.num_negotiations = 0

    def parley(self):
        """
        Alternate taking turns, until both agents have made a choice (indicated by a
        turn starting with <selection>)
        """
        if self.first_turn:
            data = self.task.episodes[self.num_negotiations % len(self.task.episodes)].strip().split()
            self.num_negotiations += 1
            for agent, tag in zip(self.agents, ['input', 'partner_input']):
                book_cnt, book_val, hat_cnt, hat_val, ball_cnt, ball_val = get_tag(data, tag)
                action = {}
                action['text'] = WELCOME_MESSAGE.format(book_cnt=book_cnt, book_val=book_val, hat_cnt=hat_cnt, hat_val=hat_val, ball_cnt=ball_cnt, ball_val=ball_val)
                action['items'] = {'book_cnt': book_cnt, 'book_val': book_val, 'hat_cnt': hat_cnt, 'hat_val': hat_val, 'ball_cnt': ball_cnt, 'ball_val': ball_val}
                agent.observe(validate(action))
            self.first_turn = False
        else:
            self.turns += 1
            for _index, agent in enumerate(self.agents):
                if agent in self.choices:
                    continue
                try:
                    act = agent.act(timeout=None)
                except TypeError:
                    act = agent.act()
                for other_agent in self.agents:
                    if other_agent != agent:
                        other_agent.observe(validate(act))
                if act['text'].startswith('<selection>') and self.turns > 1:
                    self.choices[agent] = act['text']
                    self.selection = True
                    if len(self.choices) == len(self.agents):
                        self.first_turn = True
                        self.episodeDone = True
                elif act['episode_done']:
                    self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        """
        Shutdown all mturk agents in parallel, otherwise if one mturk agent is
        disconnected then it could prevent other mturk agents from completing.
        """
        global shutdown_agent

        def shutdown_agent(agent):
            try:
                agent.shutdown(timeout=None)
            except Exception:
                agent.shutdown()
        Parallel(n_jobs=len(self.agents), backend='threading')((delayed(shutdown_agent)(agent) for agent in self.agents))