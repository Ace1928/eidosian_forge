import math
def CherkasskyRiskBound(VCDim, nData, nWrong, conf, a1=1.0, a2=2.0):
    """

    The formulation here is from Eqns 4.22 and 4.23 on pg 108 of
    Cherkassky and Mulier's book "Learning From Data" Wiley, 1998.

    **Arguments**

      - VCDim: the VC dimension of the system

      - nData: the number of data points used

      - nWrong: the number of data points misclassified

      - conf: the confidence to be used for this risk bound

      - a1, a2: constants in the risk equation. Restrictions on these values:

          - 0 <= a1 <= 4

          - 0 <= a2 <= 2

    **Returns**

      - a float


    **Notes**

     - This appears to behave reasonably

     - the equality a1=1.0 is by analogy to Burges's paper.

  """
    h = VCDim
    n = nData
    eta = conf
    rEmp = float(nWrong) / nData
    numerator = h * (math.log(float(a2 * n) / h) + 1) - math.log(eta / 4.0)
    eps = a1 * numerator / n
    structRisk = eps / 2.0 * (1.0 + math.sqrt(1.0 + 4.0 * rEmp / eps))
    return rEmp + structRisk