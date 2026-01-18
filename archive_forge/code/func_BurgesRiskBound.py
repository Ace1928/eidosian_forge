import math
def BurgesRiskBound(VCDim, nData, nWrong, conf):
    """ Calculates Burges's formulation of the risk bound

    The formulation is from Eqn. 3 of Burges's review
    article "A Tutorial on Support Vector Machines for Pattern Recognition"
     In _Data Mining and Knowledge Discovery_ Kluwer Academic Publishers
     (1998) Vol. 2

    **Arguments**

      - VCDim: the VC dimension of the system

      - nData: the number of data points used

      - nWrong: the number of data points misclassified

      - conf: the confidence to be used for this risk bound


    **Returns**

      - a float

    **Notes**

     - This has been validated against the Burges paper

     - I believe that this is only technically valid for binary classification

  """
    h = VCDim
    l = nData
    eta = conf
    numerator = h * (math.log(2.0 * l / h) + 1.0) - math.log(eta / 4.0)
    structRisk = math.sqrt(numerator / l)
    rEmp = float(nWrong) / l
    return rEmp + structRisk