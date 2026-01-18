import time
from xml.etree.ElementTree import Element, ElementTree, SubElement
 generates XML for a package that follows the RD_Model.dtd

    If provided, dataPerformance should be a sequence of 2-tuples:
      ( note, performance )
    where performance is of the form:
      ( accuracy, avgCorrectConf, avgIncorrectConf, confusionMatrix, thresh, avgSkipConf, nSkipped )
      the last four elements are optional

    