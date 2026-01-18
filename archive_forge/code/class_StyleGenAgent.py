from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from projects.style_gen.modules import StyleAgentMixin
class StyleGenAgent(StyleAgentMixin, TransformerGeneratorAgent):
    """
    General purpose generator with a style in the history.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        StyleAgentMixin.add_cmdline_args(argparser)
        TransformerGeneratorAgent.add_cmdline_args(argparser)
        return argparser