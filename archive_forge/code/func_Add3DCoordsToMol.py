import os
import tempfile
import time
import re
def Add3DCoordsToMol(data, format, props={}):
    """ adds 3D coordinates to the data passed in using Chem3D

    **Arguments**

      - data: the molecular data

      - format: the format of _data_.  Should be something accepted by
        _CDXConvert_

      - props: (optional) a dictionary used to return calculated properties
  
  """
    global c3dApp
    if c3dApp is None:
        StartChem3D()
    if format != 'chemical/mdl-molfile':
        molData = CDXClean(data, format, 'chemical/mdl-molfile')
    else:
        molData = data
    with tempfile.NamedTemporaryFile(suffix='.mol', delete=False) as molF:
        molF.write(molData)
    doc = c3dApp.Documents.Open(molF.name)
    if not doc:
        print('cannot open molecule')
        raise ValueError('No Molecule')
    job = Dispatch('Chem3D.MM2Job')
    job.Type = 1
    job.DisplayEveryIteration = 0
    job.RecordEveryIteration = 0
    doc.MM2Compute(job)
    while doc.ComputeStatus in [1129270608, 1346719300]:
        pass
    outFName = os.getcwd() + '/to3d.mol'
    doc.SaveAs(outFName)
    for prop in availChem3DProps:
        props[prop] = eval('doc.%s' % prop)
    doc.Close(0)
    os.unlink(molF.name)
    c3dData = open(outFName, 'r').read()
    gone = 0
    while not gone:
        try:
            os.unlink(outFName)
        except Exception:
            time.sleep(0.5)
        else:
            gone = 1
    return c3dData