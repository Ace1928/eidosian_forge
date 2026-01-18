import warnings
import json
import csv
import numpy as np
from Bio import BiopythonParserWarning
def CsvIterator(handle):
    """Iterate over PM csv records as PlateRecord objects.

    Arguments:
     - handle - input file

    """
    plate = None
    data = False
    qualifiers = {}
    idx = {}
    wells = {}
    tblreader = csv.reader(handle, delimiter=',', quotechar='"')
    for line in tblreader:
        if len(line) < 2:
            continue
        elif _datafile in line[0].strip():
            if plate is not None:
                qualifiers[_csvData][_datafile] = line[1].strip()
                plate = PlateRecord(plate.id)
                for k, v in wells.items():
                    plate[k] = WellRecord(k, plate, v)
                plate.qualifiers = qualifiers
                yield plate
            plate = PlateRecord(None)
            data = False
            qualifiers[_csvData] = {}
            idx = {}
            wells = {}
        elif _plate in line[0].strip():
            plateID = line[1].strip()
            qualifiers[_csvData][_plate] = plateID
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
            plate.id = plateID
        elif _strainType in line[0].strip():
            if plate is None:
                continue
            qualifiers[_csvData][_strainType] = line[1].strip()
        elif _sample in line[0].strip():
            if plate is None:
                continue
            qualifiers[_csvData][_sample] = line[1].strip()
        elif _strainNumber in line[0].strip():
            if plate is None:
                continue
            qualifiers[_csvData][_strainNumber] = line[1].strip()
        elif _strainName in line[0].strip():
            if plate is None:
                continue
            qualifiers[_csvData][_strainName] = line[1].strip()
        elif _other in line[0].strip():
            if plate is None:
                continue
            qualifiers[_csvData][_other] = line[1].strip()
        elif _file in line[0].strip():
            if plate is None:
                continue
            qualifiers[_csvData][_file] = line[1].strip()
        elif _position in line[0].strip():
            if plate is None:
                continue
            qualifiers[_csvData][_position] = line[1].strip()
        elif _setupTime in line[0].strip():
            if plate is None:
                continue
            qualifiers[_csvData][_setupTime] = line[1].strip()
        elif _hour in line[0].strip():
            if plate is None:
                continue
            data = True
            for i in range(1, len(line)):
                x = line[i]
                if x == '':
                    continue
                wells[x.strip()] = {}
                idx[i] = x.strip()
        elif data:
            if plate is None:
                continue
            try:
                float(line[0])
            except ValueError:
                continue
            time = float(line[0])
            for i in range(1, len(line)):
                x = line[i]
                try:
                    signal = float(x)
                except ValueError:
                    continue
                well = idx[i]
                wells[well][time] = signal
    if plate is not None and plate.id is not None:
        plate = PlateRecord(plate.id)
        for k, v in wells.items():
            plate[k] = WellRecord(k, plate, v)
        plate.qualifiers = qualifiers
        yield plate