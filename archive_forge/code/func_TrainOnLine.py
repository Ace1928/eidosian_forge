import numpy
def TrainOnLine(self, examples, net, maxIts=5000, errTol=0.1, useAvgErr=1, silent=0):
    """ carries out online training of a neural net

      The definition of online training is that the network is updated after
        each example is presented.

      **Arguments**

        - examples: a list of 2-tuple:
           1) a list of variable values values
           2) a list of result values (targets)

        - net: a _Network_ (or something supporting the same API)

        - maxIts: the maximum number of *training epochs* (see below for definition) to be
          run

        - errTol: the tolerance for convergence

        - useAvgErr: if this toggle is nonzero, then the error at each step will be
          divided by the number of training examples for the purposes of checking
          convergence.

        - silent: controls the amount of visual noise produced as this runs.


      **Note**

         a *training epoch* is one complete pass through all the training examples

    """
    nExamples = len(examples)
    converged = 0
    cycle = 0
    while not converged and cycle < maxIts:
        maxErr = 0
        newErr = 0
        for example in examples:
            localErr = self.StepUpdate(example, net)
            newErr += localErr
            if localErr > maxErr:
                maxErr = localErr
        if useAvgErr == 1:
            newErr = newErr / nExamples
        else:
            newErr = maxErr
        if newErr <= errTol:
            converged = 1
        if not silent:
            print('epoch %d, error: % 6.4f' % (cycle, newErr))
        cycle = cycle + 1
    if not silent:
        if converged:
            print('Converged after %d epochs.' % cycle)
        else:
            print('NOT Converged after %d epochs.' % cycle)
        print('final error: % 6.4f' % newErr)