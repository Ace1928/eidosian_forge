from parlai.core.worlds import World
from parlai.mturk.core.dev.agents import AssignState
class MTurkTaskWorld(MTurkDataWorld):
    """
    Generic world for MTurk tasks.
    """

    def __init__(self, opt, mturk_agent):
        """
        Init should set up resources for running the task world.
        """
        self.mturk_agent = mturk_agent
        self.episodeDone = False

    def parley(self):
        """
        A parley should represent one turn of your task.
        """
        self.episodeDone = True

    def episode_done(self):
        """
        A ParlAI-MTurk task ends and allows workers to be marked complete when the world
        is finished.
        """
        return self.episodeDone

    def shutdown(self):
        """
        Should be used to free the world's resources and shut down the agents.

        Use the following code if there are multiple MTurk agents:

        global shutdown_agent
        def shutdown_agent(mturk_agent):
            mturk_agent.shutdown()
        Parallel(
            n_jobs=len(self.mturk_agents),
            backend='threading'
        )(delayed(shutdown_agent)(agent) for agent in self.mturk_agents)
        """
        self.mturk_agent.shutdown()

    def review_work(self):
        """
        Programmatically approve/reject the turker's work. Doing this now (if possible)
        means that you don't need to do the work of reviewing later on.

        For example:
        .. code-block:: python
            if self.turker_response == '0':
                self.mturk_agent.reject_work(
                    'You rated our model's response as a 0/10 but we '
                    'know we're better than that'
                )
            else:
                if self.turker_response == '10':
                    self.mturk_agent.pay_bonus(1, 'Thanks for a great rating!')
                self.mturk_agent.approve_work()
        """
        pass