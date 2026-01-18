import abc
class WorkerBase(object):

    @property
    def _workers(self):
        try:
            return self.__workers
        except AttributeError:
            self.__workers = []
        return self.__workers

    def get_workers(self):
        """Returns a collection NeutronWorker instances needed by this service.

        """
        return list(self._workers)

    def add_worker(self, worker):
        """Adds NeutronWorker needed for this service

        If a object needs to define workers thread/processes outside of API/RPC
        workers then it will call this method to register worker. Should be
        called on initialization stage before running services
        """
        self._workers.append(worker)

    def add_workers(self, workers):
        """Adds NeutronWorker list needed for this service

        The same as add_worker but adds a list of workers
        """
        self._workers.extend(workers)