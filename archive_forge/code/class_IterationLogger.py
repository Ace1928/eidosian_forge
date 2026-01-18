import logging
class IterationLogger:
    """
    Log (and print) information for all iterations
    """

    def __init__(self):
        self.iterations = []

    def newIteration(self, iteration, feasibility, objectiveValue, trustRadius, stepNorm):
        """
        Add a new iteration to the list of iterations
        """
        self.iterrecord = IterationRecord(iteration, feasibility=feasibility, objectiveValue=objectiveValue, trustRadius=trustRadius, stepNorm=stepNorm)
        self.iterations.append(self.iterrecord)

    def updateIteration(self, feasibility=None, objectiveValue=None, trustRadius=None, stepNorm=None):
        """
        Update values in current record
        """
        if feasibility is not None:
            self.iterrecord.feasibility = feasibility
        if objectiveValue is not None:
            self.iterrecord.objectiveValue = objectiveValue
        if trustRadius is not None:
            self.iterrecord.trustRadius = trustRadius
        if stepNorm is not None:
            self.iterrecord.stepNorm = stepNorm

    def logIteration(self):
        """
        Log detailed information about the iteration to the log
        """
        self.iterrecord.detailLogger()

    def printIteration(self):
        """
        Print information to the screen
        """
        self.iterrecord.verboseLogger()