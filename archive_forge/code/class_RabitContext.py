import logging
import xgboost as xgb
class RabitContext:
    """
    Context to connect a worker to a rabit tracker.

    Parameters
    ----------
    actor_rank : int
        Rank of actor, connected to this context.
    args : list
        List with environment variables for Rabit Tracker.
    """

    def __init__(self, actor_rank, args):
        self.args = args
        self.args.append(('DMLC_TASK_ID=[modin.xgboost]:' + str(actor_rank)).encode())

    def __enter__(self):
        """
        Entry point of context.

        Connects to Rabit Tracker.
        """
        xgb.rabit.init(self.args)
        LOGGER.info('-------------- rabit started ------------------')

    def __exit__(self, *args):
        """
        Exit point of context.

        Disconnects from Rabit Tracker.

        Parameters
        ----------
        *args : iterable
            Parameters for Exception capturing.
        """
        xgb.rabit.finalize()
        LOGGER.info('-------------- rabit finished ------------------')