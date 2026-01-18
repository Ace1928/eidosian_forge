from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.utils.safety import OffensiveStringMatcher, OffensiveLanguageClassifier
from parlai.utils.misc import TimeLogger
import parlai.utils.logging as logging
from parlai.core.script import ParlaiScript, register_script
@register_script('detect_offensive', hidden=True)
class DetectOffensive(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return detect(self.opt)