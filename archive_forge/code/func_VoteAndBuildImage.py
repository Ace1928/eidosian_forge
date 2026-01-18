import numpy
from PIL import Image, ImageDraw
def VoteAndBuildImage(composite, data, badOnly=0, sortTrueVals=0, xScale=10, yScale=2, addLine=1):
    """ collects votes on the examples and constructs an image

    **Arguments**

      - composte: a composite model

      - data: the examples to be voted upon

      - badOnly: if nonzero only the incorrect votes will be shown

      - sortTrueVals: if nonzero the votes will be sorted so
        that the _trueValues_ are in order, otherwise the sort
        is by _values_

      - xScale: number of pixels per vote in the x direction

      - yScale: number of pixels per example in the y direction

      - addLine: if nonzero, a purple line is drawn separating
         the votes from the examples

    **Returns**

      a PIL image

  """
    nModels = len(composite) + 3
    print('nModels:', nModels - 3)
    res, values, trueValues, misCount = CollectVotes(composite, data, badOnly)
    print('%d examples were misclassified' % misCount)
    img = BuildVoteImage(nModels, res, values, trueValues, sortTrueVals, xScale, yScale, addLine)
    return img