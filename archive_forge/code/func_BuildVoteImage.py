import numpy
from PIL import Image, ImageDraw
def BuildVoteImage(nModels, data, values, trueValues=[], sortTrueVals=0, xScale=10, yScale=2, addLine=1):
    """ constructs the actual image

    **Arguments**

      - nModels: the number of models in the composite

      - data: the results of voting

      - values: predicted values for each example

      - trueValues: true values for each example

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
    nData = len(data)
    data = numpy.array(data, numpy.integer)
    if sortTrueVals and trueValues != []:
        order = numpy.argsort(trueValues)
    else:
        order = numpy.argsort(values)
    data = [data[x] for x in order]
    maxVal = max(numpy.ravel(data))
    data = data * 255 / maxVal
    datab = data.astype('B')
    img = Image.frombytes('L', (nModels, nData), datab.tobytes())
    if addLine:
        img = img.convert('RGB')
        canvas = ImageDraw.Draw(img)
        if trueValues != []:
            canvas.line([(nModels - 3, 0), (nModels - 3, nData)], fill=(128, 0, 128))
        else:
            canvas.line([(nModels - 2, 0), (nModels - 2, nData)], fill=(128, 0, 128))
    img = img.resize((nModels * xScale, nData * yScale))
    return img