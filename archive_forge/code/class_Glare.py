import itertools
import math
import operator
import random
from functools import reduce
class Glare:
    """Glare Algorithm.  Implementation of

    GLARE: A New Approach for Filtering Large Reagent Lists in 
           Combinatorial Library Design Using Product Properties
    Jean-Francois Truchon* and Christopher I. Bayly

    http://pubs.acs.org/doi/pdf/10.1021/ci0504871

    Usage:
       # somehow make sidechains1/2 with props [mw, alogp, tpsa]
       r1 = RGroups(sidechains1)
       r2 = RGroups(sidechains2)
       lib = Library([r1, r2])
       props = [
         Property("mw", 0, 0, 500),
         Property("alogp", 1, -2.4, 5),
         Property("tpsa", 2, 0, 90)
       ] 

      glare = Glare()
      glare.optimize(lib, props)
    """

    def __init__(self, desiredFinalGoodness=0.95, maxIterations=100, rgroupScale=6.0, initialFraction=None, numPartitions=16):
        self.fractionGood = self.desiredFinalGoodness = desiredFinalGoodness
        self.maxIterations = maxIterations
        self.rgroupScale = rgroupScale
        if initialFraction is not None:
            self.initialFraction = initialFraction / 100.0
        else:
            self.initialFraction = initialFraction
        self.numPartitions = numPartitions

    def optimize(self, library, props):
        """library, props
        Given a Library and the list of Propery evaluators,
        optimize the library.
        The library is modified in place by removing building blocks
        (sidechains) that are not likely to pass the property
        criteria.
        """
        print('------- PARAMETERS: --------------')
        print('GOOODNESS THRESHOLD : %s%%' % (self.desiredFinalGoodness * 100))
        print('MIN PARTITION SIZE : %s' % self.numPartitions)
        if self.initialFraction is None or self.initialFraction > 0.999:
            print('INITIAL FRACTION TO KEEP : AUTOMATIC')
        else:
            print('INITIAL FRACTION TO KEEP : %s%%' % (self.initialFraction * 100))
        print('Actual SIZE : %s = %s' % (' x '.join([str(len(rg.sidechains)) for rg in library.rgroups]), reduce(operator.mul, [len(rg.sidechains) for rg in library.rgroups])))
        running_total = 0.0
        Gt = self.desiredFinalGoodness
        for iteration in range(1, self.maxIterations + 1):
            for rg in library.rgroups:
                rg.randomize()
            good = total = 0.0
            chunked_libs = library.chunk(self.numPartitions)
            for libidx, chunk in enumerate(chunked_libs):
                g, t = chunk.evaluate(props)
                good += g
                total += t
            running_total += total
            Gi = good / total
            if Gi < 1e-12:
                fraction = 0.0
            elif iteration == 1:
                G0 = Gi
                if self.initialFraction is not None:
                    fraction = K0 = self.initialFraction
                else:
                    fraction = K0 = min(-1.1 * (Gt - G0) + 1.2, 0.9)
            else:
                if abs(Gt - G0) < 0.0001:
                    Ki = 1.0
                else:
                    Ki = (1.0 - K0) * (Gi - G0) / (Gt - G0) + K0
                fraction = min(1.0, Ki)
            max_size = float(max([len(rg.sidechains) for rg in library.rgroups]))
            for rg in library.rgroups:
                scale = 1.0
                if self.rgroupScale is not None:
                    numSidechains = len(rg.sidechains)
                    numer = 1.0
                    denom = 1.0 + math.exp(-self.rgroupScale * (numSidechains / max_size - 0.5))
                    scale = numer / denom
                fraction_to_reject = (1.0 - fraction) * scale
                rg.prune(1.0 - fraction_to_reject)
            print('-------------- ITERATION : %s ----------------------' % iteration)
            print('GOODNESS      : %s%%' % (Gi * 100))
            print('NUMBER EVAL   : %s' % total)
            print('CUMUL EVAL    : %s' % running_total)
            print('KEPT IN STEP  : %s%%' % (fraction * 100.0))
            if not iteration:
                print('GOODNESS THRESHOLD : %s' % self.desiredFinalGoodness)
                print('MIN PARTITION SIZE : %s' % self.numPartitions)
                print('INITIAL FRACTION TO KEEP : ')
                if self.fractionToKeep > 0.999:
                    print('AUTOMATIC')
                else:
                    print('%s%%' % self.fractionToKeep)
            print('Actual SIZE : %s = %s' % (' x '.join([str(len(rg.sidechains)) for rg in library.rgroups]), reduce(operator.mul, [len(rg.sidechains) for rg in library.rgroups])))
            print('EFFECTIVENESS : %s%%' % (library.effectiveness() * 100.0))
            if iteration and Gi < 1e-12:
                return
            elif abs(Gi - self.desiredFinalGoodness) < 0.001 or Gi > self.desiredFinalGoodness:
                return