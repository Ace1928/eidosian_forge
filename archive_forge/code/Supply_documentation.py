from rdkit.VLib.Node import VLibNode
 base class for nodes which supply things

    Assumptions:
      1) no parents

    Usage Example:
    
      >>> supplier = SupplyNode(contents=[1,2,3])
      >>> supplier.next()
      1
      >>> supplier.next()
      2
      >>> supplier.next()
      3
      >>> supplier.next()
      Traceback (most recent call last):
          ...
      StopIteration
      >>> supplier.reset()
      >>> supplier.next()
      1
      >>> [x for x in supplier]
      [1, 2, 3]


    