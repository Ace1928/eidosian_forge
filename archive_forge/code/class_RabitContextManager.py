import logging
import xgboost as xgb
class RabitContextManager:
    """
    A manager class that controls lifecycle of `xgb.RabitTracker`.

    All workers that are used for distributed training will connect to
    Rabit Tracker stored in this class.

    Parameters
    ----------
    num_workers : int
        Number of workers of `self.rabit_tracker`.
    host_ip : str
        IP address of host that creates `self` object.
    """

    def __init__(self, num_workers: int, host_ip):
        self._num_workers = num_workers
        self.env = {'DMLC_NUM_WORKER': self._num_workers}
        self.rabit_tracker = xgb.RabitTracker(host_ip=host_ip, n_workers=self._num_workers)

    def __enter__(self):
        """
        Entry point of manager.

        Updates Rabit Tracker environment, starts `self.rabit_tracker`.

        Returns
        -------
        dict
            Dict with Rabit Tracker environment.
        """
        self.env.update(self.rabit_tracker.worker_envs())
        self.rabit_tracker.start(self._num_workers)
        return self.env

    def __exit__(self, type, value, traceback):
        """
        Exit point of manager.

        Finishes `self.rabit_tracker`.

        Parameters
        ----------
        type : exception type
            Type of exception, captured  by manager.
        value : Exception
            Exception value.
        traceback : TracebackType
            Traceback of exception.
        """
        self.rabit_tracker.join()