import random
from typing import Optional
from parlai.core.message import Message
from parlai.core.opt import Opt
class StyleAgentMixin:
    """
    Methods for agents that return style from their histories.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.

        Does not add arguments from its superclass because it's a mixin.
        """
        agent = argparser.add_argument_group('StyleAgentMixin arguments')
        agent.add_argument('--use-style-frac', type=float, default=1.0, help='What fraction of the time to use the style label')
        return agent

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        self.use_style_frac = opt['use_style_frac']

    def get_temp_history(self, observation: Message) -> Optional[str]:
        """
        Conditionally return a style-token string to temporarily insert into history.
        """
        use_style_rand = random.random()
        if use_style_rand < self.use_style_frac:
            style = observation.get('personality')
        else:
            style = ''
        if style is not None and style != '':
            return STYLE_SEP_TOKEN + style