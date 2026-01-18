from rdkit.ML.Data import Stats
def StdDev(mat):
    """ the standard deviation classifier

   This uses _ML.Data.Stats.StandardizeMatrix()_ to do the work

  """
    return Stats.StandardizeMatrix(mat)