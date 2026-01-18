from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
from parlai.core.worlds import validate
from joblib import Parallel, delayed
class MTurkMultiAgentDialogOnboardWorld(MTurkOnboardWorld):

    def parley(self):
        self.mturk_agent.observe({'id': 'System', 'text': 'Welcome onboard!'})
        self.mturk_agent.act()
        self.mturk_agent.observe({'id': 'System', 'text': 'Thank you for your input! Please wait while we match you with another worker...'})
        self.episodeDone = True