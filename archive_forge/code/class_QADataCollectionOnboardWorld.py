from parlai.core.worlds import validate
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
class QADataCollectionOnboardWorld(MTurkOnboardWorld):
    """
    Example onboarding world.

    Sends a message from the world to the worker and then exits as complete
    """

    def parley(self):
        ad = {}
        ad['id'] = 'System'
        ad['text'] = 'Welcome onboard!'
        self.mturk_agent.observe(ad)
        self.mturk_agent.act()
        self.episodeDone = True