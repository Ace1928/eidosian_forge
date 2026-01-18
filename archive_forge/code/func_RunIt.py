from warnings import warn
import pickle
import sys
import time
import numpy
from rdkit import DataStructs
from rdkit.Dbase import DbModule
from rdkit.ML import CompositeRun, ScreenComposite
from rdkit.ML.Composite import BayesComposite, Composite
from rdkit.ML.Data import DataUtils, SplitData
from rdkit.utils import listutils
def RunIt(details, progressCallback=None, saveIt=1, setDescNames=0):
    """ does the actual work of building a composite model

    **Arguments**

      - details:  a _CompositeRun.CompositeRun_ object containing details
        (options, parameters, etc.) about the run

      - progressCallback: (optional) a function which is called with a single
        argument (the number of models built so far) after each model is built.

      - saveIt: (optional) if this is nonzero, the resulting model will be pickled
        and dumped to the filename specified in _details.outName_

      - setDescNames: (optional) if nonzero, the composite's _SetInputOrder()_ method
        will be called using the results of the data set's _GetVarNames()_ method;
        it is assumed that the details object has a _descNames attribute which
        is passed to the composites _SetDescriptorNames()_ method.  Otherwise
        (the default), _SetDescriptorNames()_ gets the results of _GetVarNames()_.

    **Returns**

      the composite model constructed


  """
    details.rundate = time.asctime()
    fName = details.tableName.strip()
    if details.outName == '':
        details.outName = fName + '.pkl'
    if not details.dbName:
        if details.qBounds != []:
            data = DataUtils.TextFileToData(fName)
        else:
            data = DataUtils.BuildQuantDataSet(fName)
    elif details.useSigTrees or details.useSigBayes:
        details.tableName = fName
        data = details.GetDataSet(pickleCol=0, pickleClass=DataStructs.ExplicitBitVect)
    elif details.qBounds != [] or not details.useTrees:
        details.tableName = fName
        data = details.GetDataSet()
    else:
        data = DataUtils.DBToQuantData(details.dbName, fName, quantName=details.qTableName, user=details.dbUser, password=details.dbPassword)
    composite = RunOnData(details, data, progressCallback=progressCallback, saveIt=saveIt, setDescNames=setDescNames)
    return composite