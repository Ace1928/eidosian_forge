from parlai.core.worlds import validate
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
import parlai.mturk.core.mturk_utils as mturk_utils
import random
class QualificationFlowOnboardWorld(MTurkOnboardWorld):

    def parley(self):
        ad = {}
        ad['id'] = 'System'
        ad['text'] = 'This demo displays the functionality of using qualifications to filter the workers who are able to do your tasks. The first task you will get will check to see if you pass the bar that the task requires against a prepared test set. If you pass, the next task will be a real one rather than the test one.\nSend anything to get started.'
        self.mturk_agent.observe(ad)
        self.mturk_agent.act()
        self.episodeDone = True