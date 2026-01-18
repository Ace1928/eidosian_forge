import math
import re
from nltk.tokenize.api import TokenizerI
def _identify_boundaries(self, depth_scores):
    """Identifies boundaries at the peaks of similarity score
        differences"""
    boundaries = [0 for x in depth_scores]
    avg = sum(depth_scores) / len(depth_scores)
    stdev = numpy.std(depth_scores)
    if self.cutoff_policy == LC:
        cutoff = avg - stdev
    else:
        cutoff = avg - stdev / 2.0
    depth_tuples = sorted(zip(depth_scores, range(len(depth_scores))))
    depth_tuples.reverse()
    hp = list(filter(lambda x: x[0] > cutoff, depth_tuples))
    for dt in hp:
        boundaries[dt[1]] = 1
        for dt2 in hp:
            if dt[1] != dt2[1] and abs(dt2[1] - dt[1]) < 4 and (boundaries[dt2[1]] == 1):
                boundaries[dt[1]] = 0
    return boundaries