import logging
class IterationRecord:
    """
    Record relevant information at each individual iteration
    """

    def __init__(self, iteration, feasibility=None, objectiveValue=None, trustRadius=None, stepNorm=None):
        self.iteration = iteration
        self.fStep, self.thetaStep, self.rejected = [False] * 3
        if feasibility is not None:
            self.feasibility = feasibility
        if objectiveValue is not None:
            self.objectiveValue = objectiveValue
        if trustRadius is not None:
            self.trustRadius = trustRadius
        if stepNorm is not None:
            self.stepNorm = stepNorm

    def detailLogger(self):
        """
        Print information about the iteration to the log.
        """
        logger.info('****** Iteration %d ******' % self.iteration)
        logger.info('trustRadius = %s' % self.trustRadius)
        logger.info('feasibility = %s' % self.feasibility)
        logger.info('objectiveValue = %s' % self.objectiveValue)
        logger.info('stepNorm = %s' % self.stepNorm)
        if self.fStep:
            logger.info('INFO: f-type step')
        if self.thetaStep:
            logger.info('INFO: theta-type step')
        if self.rejected:
            logger.info('INFO: step rejected')

    def verboseLogger(self):
        """
        Print information about the iteration to the console
        """
        print('****** Iteration %d ******' % self.iteration)
        print('objectiveValue = %s' % self.objectiveValue)
        print('feasibility = %s' % self.feasibility)
        print('trustRadius = %s' % self.trustRadius)
        print('stepNorm = %s' % self.stepNorm)
        if self.fStep:
            print('INFO: f-type step')
        if self.thetaStep:
            print('INFO: theta-type step')
        if self.rejected:
            print('INFO: step rejected')
        print(25 * '*')