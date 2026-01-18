from math import *
from rdkit import RDConfig
def CalcSingleCompoundDescriptor(compos, argVect, atomDict, propDict):
    """ calculates the value of the descriptor for a single compound

    **ARGUMENTS:**

      - compos: a vector/tuple containing the composition
         information... in the form:
         '[("Fe",1.),("Pt",2.),("Rh",0.02)]'

      - argVect: a vector/tuple with three elements:

           1) AtomicDescriptorNames:  a list/tuple of the names of the
             atomic descriptors being used. These determine the
             meaning of $1, $2, etc. in the expression

           2) CompoundDescriptorNames:  a list/tuple of the names of the
             compound descriptors being used. These determine the
             meaning of $a, $b, etc. in the expression

           3) Expr: a string containing the expression to be used to
             evaluate the final result.

      - atomDict:
           a dictionary of atomic descriptors.  Each atomic entry is
           another dictionary containing the individual descriptors
           and their values

      - propVect:
           a list of descriptors for the composition.

    **RETURNS:**

      the value of the descriptor, -666 if a problem was encountered

    **NOTE:**

      - because it takes rather a lot of work to get everything set
          up to calculate a descriptor, if you are calculating the
          same descriptor for multiple compounds, you probably want to
          be calling _CalcMultipleCompoundsDescriptor()_.

  """
    try:
        atomVarNames = argVect[0]
        compositionVarNames = argVect[1]
        formula = argVect[2]
        formula = _SubForCompoundDescriptors(formula, compositionVarNames, 'propDict')
        formula = _SubForAtomicVars(formula, atomVarNames, 'atomDict')
        evalTarget = _SubMethodArgs(formula, knownMethods)
    except Exception:
        if __DEBUG:
            import traceback
            print('Sub Failure!')
            traceback.print_exc()
            print(evalTarget)
            print(propDict)
            raise RuntimeError('Failure 1')
        else:
            return -666
    try:
        v = eval(evalTarget)
    except Exception:
        if __DEBUG:
            import traceback
            outF = open(RDConfig.RDCodeDir + '/ml/descriptors/log.txt', 'a+')
            outF.write('#------------------------------\n')
            outF.write('formula: %s\n' % repr(formula))
            outF.write('target: %s\n' % repr(evalTarget))
            outF.write('propDict: %s\n' % repr(propDict))
            outF.write('keys: %s\n' % repr(sorted(atomDict)))
            outF.close()
            print('ick!')
            print('formula:', formula)
            print('target:', evalTarget)
            print('propDict:', propDict)
            print('keys:', atomDict.keys())
            traceback.print_exc()
            raise RuntimeError('Failure 2')
        else:
            v = -666
    return v