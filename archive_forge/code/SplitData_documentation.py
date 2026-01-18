import random
from rdkit import RDRandom
  "splits" a data set held in a DB by returning lists of ids

  **Arguments**:

    - conn: a DbConnect object

    - frac: the split fraction. This can optionally be specified as a
      sequence with a different fraction for each activity value.

    - table,fields,where,join: (optional) SQL query parameters

    - useActs: (optional) toggles splitting based on activities
      (ensuring that a given fraction of each activity class ends
      up in the hold-out set)
      Defaults to 0

    - nActs: (optional) number of possible activity values, only
      used if _useActs_ is nonzero
      Defaults to 2

    - actCol: (optional) name of the activity column
      Defaults to use the last column returned by the query

    - actBounds: (optional) sequence of activity bounds
      (for cases where the activity isn't quantized in the db)
      Defaults to an empty sequence

    - silent: controls the amount of visual noise produced.

  **Usage**:

  Set up the db connection, the simple tables we're using have actives with even
  ids and inactives with odd ids:
  >>> from rdkit.ML.Data import DataUtils
  >>> from rdkit.Dbase.DbConnection import DbConnect
  >>> from rdkit import RDConfig
  >>> conn = DbConnect(RDConfig.RDTestDatabase)

  Pull a set of points from a simple table... take 33% of all points:
  >>> DataUtils.InitRandomNumbers((23,42))
  >>> train,test = SplitDbData(conn,1./3.,'basic_2class')
  >>> [str(x) for x in train]
  ['id-7', 'id-6', 'id-2', 'id-8']

  ...take 50% of actives and 50% of inactives:
  >>> DataUtils.InitRandomNumbers((23,42))
  >>> train,test = SplitDbData(conn,.5,'basic_2class',useActs=1)
  >>> [str(x) for x in train]
  ['id-5', 'id-3', 'id-1', 'id-4', 'id-10', 'id-8']


  Notice how the results came out sorted by activity

  We can be asymmetrical: take 33% of actives and 50% of inactives:
  >>> DataUtils.InitRandomNumbers((23,42))
  >>> train,test = SplitDbData(conn,[.5,1./3.],'basic_2class',useActs=1)
  >>> [str(x) for x in train]
  ['id-5', 'id-3', 'id-1', 'id-4', 'id-10']

  And we can pull from tables with non-quantized activities by providing
  activity quantization bounds:
  >>> DataUtils.InitRandomNumbers((23,42))
  >>> train,test = SplitDbData(conn,.5,'float_2class',useActs=1,actBounds=[1.0])
  >>> [str(x) for x in train]
  ['id-5', 'id-3', 'id-1', 'id-4', 'id-10', 'id-8']

  