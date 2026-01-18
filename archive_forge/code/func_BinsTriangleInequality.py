import itertools
def BinsTriangleInequality(d1, d2, d3):
    """ checks the triangle inequality for combinations
      of distance bins.

      the general triangle inequality is:
         d1 + d2 >= d3
      the conservative binned form of this is:
         d1(upper) + d2(upper) >= d3(lower)

    """
    if d1[1] + d2[1] < d3[0]:
        return False
    if d2[1] + d3[1] < d1[0]:
        return False
    if d3[1] + d1[1] < d2[0]:
        return False
    return True