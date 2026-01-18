def GetAvailTransforms():
    """ returns the list of available data transformations

   **Returns**

     a list of 3-tuples

       1) name of the transform (text)

       2) function describing the transform (should take an
          _MLDataSet_ as an argument)

       3) description of the transform (text)

  """
    global _availTransforms
    return _availTransforms