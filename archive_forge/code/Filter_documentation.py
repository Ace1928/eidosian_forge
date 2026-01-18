from rdkit.VLib.Node import VLibNode
 base class for nodes which filter their input

    Assumptions:

      - filter function takes a number of arguments equal to the
        number of inputs we have.  It returns a bool

      - inputs (parents) can be stepped through in lockstep

      - we return a tuple if there's more than one input

    Usage Example:

      >>> from rdkit.VLib.Supply import SupplyNode
      >>> def func(a,b):
      ...   return a+b < 5
      >>> filt = FilterNode(func=func)
      >>> suppl1 = SupplyNode(contents=[1,2,3,3])
      >>> suppl2 = SupplyNode(contents=[1,2,3,1])
      >>> filt.AddParent(suppl1)
      >>> filt.AddParent(suppl2)
      >>> v = [x for x in filt]
      >>> v
      [(1, 1), (2, 2), (3, 1)]
      >>> filt.reset()
      >>> v = [x for x in filt]
      >>> v
      [(1, 1), (2, 2), (3, 1)]
      >>> filt.Destroy()

      Negation is also possible:

      >>> filt = FilterNode(func=func,negate=1)
      >>> suppl1 = SupplyNode(contents=[1,2,3,3])
      >>> suppl2 = SupplyNode(contents=[1,2,3,1])
      >>> filt.AddParent(suppl1)
      >>> filt.AddParent(suppl2)
      >>> v = [x for x in filt]
      >>> v
      [(3, 3)]
      >>> filt.Destroy()

      With no function, just return the inputs:

      >>> filt = FilterNode()
      >>> suppl1 = SupplyNode(contents=[1,2,3,3])
      >>> filt.AddParent(suppl1)
      >>> v = [x for x in filt]
      >>> v
      [1, 2, 3, 3]
      >>> filt.Destroy()

    