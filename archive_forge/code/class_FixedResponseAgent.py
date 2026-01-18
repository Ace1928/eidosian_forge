from parlai.core.agents import Agent
from parlai.core.message import Message
class FixedResponseAgent(Agent):

    def add_cmdline_args(argparser):
        group = argparser.add_argument_group('FixedResponse Arguments')
        group.add_argument('-fr', '--fixed-response', type='nonestr', default="I don't know.", help='fixed response the agent always returns')

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'FixedResponseAgent'
        self.fixed_response = self.opt['fixed_response']

    def act(self):
        return Message({'id': self.getID(), 'text': self.fixed_response, 'episode_done': False})