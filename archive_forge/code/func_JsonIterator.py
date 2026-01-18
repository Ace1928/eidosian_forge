import warnings
import json
import csv
import numpy as np
from Bio import BiopythonParserWarning
def JsonIterator(handle):
    """Iterate over PM json records as PlateRecord objects.

    Arguments:
     - handle - input file

    """
    try:
        data = json.load(handle)
    except ValueError:
        raise ValueError('Could not parse JSON file')
    if hasattr(data, 'keys'):
        data = [data]
    for pobj in data:
        try:
            plateID = pobj[_csvData][_plate]
        except TypeError:
            raise TypeError('Malformed JSON input')
        except KeyError:
            raise KeyError('Could not retrieve plate id')
        if not plateID.startswith(_platesPrefix) and (not plateID.startswith(_platesPrefixMammalian)):
            warnings.warn(f'Non-standard plate ID found ({plateID})', BiopythonParserWarning)
        else:
            if plateID.startswith(_platesPrefixMammalian):
                pID = plateID[len(_platesPrefixMammalian):]
            else:
                pID = plateID[len(_platesPrefix):]
            while len(pID) > 0:
                try:
                    int(pID)
                    break
                except ValueError:
                    pID = pID[:-1]
            if len(pID) == 0:
                warnings.warn(f'Non-standard plate ID found ({plateID})', BiopythonParserWarning)
            elif int(pID) < 0:
                warnings.warn(f'Non-standard plate ID found ({plateID}), using {_platesPrefix}{abs(int(pID))}')
                plateID = _platesPrefix + str(abs(int(pID)))
            elif plateID.startswith(_platesPrefixMammalian):
                plateID = _platesPrefixMammalian + '%02d' % int(pID)
            else:
                plateID = _platesPrefix + '%02d' % int(pID)
        try:
            times = pobj[_measurements][_hour]
        except KeyError:
            raise KeyError('Could not retrieve the time points')
        plate = PlateRecord(plateID)
        for k in pobj[_measurements]:
            if k == _hour:
                continue
            plate[k] = WellRecord(k, plate=plate, signals={times[i]: pobj[_measurements][k][i] for i in range(len(times))})
        del pobj['measurements']
        plate.qualifiers = pobj
        yield plate