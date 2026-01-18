from rdkit import RDConfig
from rdkit.Dbase import DbModule
def ValidateRDId(ID):
    """ returns whether or not an RDId is valid

  >>> ValidateRDId('RDCmpd-000-009-9')
  1
  >>> ValidateRDId('RDCmpd-009-000-009-8')
  1
  >>> ValidateRDId('RDCmpd-009-000-109-8')
  0
  >>> ValidateRDId('bogus')
  0

  """
    ID = ID.replace('_', '-')
    splitId = ID.split('-')
    if len(splitId) < 4:
        return 0
    accum = 0
    for entry in splitId[1:-1]:
        for char in entry:
            try:
                v = int(char)
            except ValueError:
                return 0
            accum += v
    crc = int(splitId[-1])
    return accum % 10 == crc