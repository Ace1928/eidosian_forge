import time
from xml.etree.ElementTree import Element, ElementTree, SubElement
def _ConvertModelPerformance(perf, modelPerf):
    if len(modelPerf) > 3:
        confMat = modelPerf[3]
        accum = 0
        for row in confMat:
            for entry in row:
                accum += entry
        accum = str(accum)
    else:
        confMat = None
        accum = 'N/A'
    if len(modelPerf) > 4:
        elem = SubElement(perf, 'ScreenThreshold')
        elem.text = str(modelPerf[4])
    elem = SubElement(perf, 'NumScreened')
    elem.text = accum
    if len(modelPerf) > 4:
        elem = SubElement(perf, 'NumSkipped')
        elem.text = str(modelPerf[6])
    elem = SubElement(perf, 'Accuracy')
    elem.text = str(modelPerf[0])
    elem = SubElement(perf, 'AvgCorrectConf')
    elem.text = str(modelPerf[1])
    elem = SubElement(perf, 'AvgIncorrectConf')
    elem.text = str(modelPerf[2])
    if len(modelPerf) > 4:
        elem = SubElement(perf, 'AvgSkipConf')
        elem.text = str(modelPerf[5])
    if confMat:
        elem = SubElement(perf, 'ConfusionMatrix')
        elem.text = str(confMat)