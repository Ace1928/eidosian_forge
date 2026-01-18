from parlai.core.message import Message
from parlai.utils.misc import display_messages
from parlai.utils.strings import colorize
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.utils.safety import OffensiveStringMatcher, OffensiveLanguageClassifier
def _init_safety(self, opt):
    """
        Initialize safety modules.
        """
    if opt['safety'] == 'string_matcher' or opt['safety'] == 'all':
        self.offensive_string_matcher = OffensiveStringMatcher()
    if opt['safety'] == 'classifier' or opt['safety'] == 'all':
        self.offensive_classifier = OffensiveLanguageClassifier()
    self.self_offensive = False