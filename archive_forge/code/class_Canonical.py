import abc
import copy
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.utilities import performance_utils as pu
from cvxpy.utilities.deterministic import unique_list
class Canonical:
    """
    An interface for objects that can be canonicalized.
    """
    __metaclass__ = abc.ABCMeta

    @property
    def expr(self):
        if not len(self.args) == 1:
            raise ValueError("'expr' is ambiguous, there should be only one argument")
        return self.args[0]

    @pu.lazyprop
    def canonical_form(self):
        """The graph implementation of the object stored as a property.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        return self.canonicalize()

    def variables(self):
        """Returns all the variables present in the arguments.
        """
        return unique_list([var for arg in self.args for var in arg.variables()])

    def parameters(self):
        """Returns all the parameters present in the arguments.
        """
        return unique_list([param for arg in self.args for param in arg.parameters()])

    def constants(self):
        """Returns all the constants present in the arguments.
        """
        return unique_list([const for arg in self.args for const in arg.constants()])

    def tree_copy(self, id_objects=None):
        new_args = []
        for arg in self.args:
            if isinstance(arg, list):
                arg_list = [elem.tree_copy(id_objects) for elem in arg]
                new_args.append(arg_list)
            else:
                new_args.append(arg.tree_copy(id_objects))
        return self.copy(args=new_args, id_objects=id_objects)

    def copy(self, args=None, id_objects=None):
        """Returns a shallow copy of the object.

        Used to reconstruct an object tree.

        Parameters
        ----------
        args : list, optional
            The arguments to reconstruct the object. If args=None, use the
            current args of the object.

        Returns
        -------
        Expression
        """
        id_objects = {} if id_objects is None else id_objects
        if id(self) in id_objects:
            return id_objects[id(self)]
        if args is None:
            args = self.args
        else:
            assert len(args) == len(self.args)
        data = self.get_data()
        if data is not None:
            return type(self)(*args + data)
        else:
            return type(self)(*args)

    def __copy__(self):
        """
        Called by copy.copy()
        Creates a shallow copy of the object, that is, the copied object refers to the same
        leaf nodes as the original object. Non-leaf nodes are recreated.
        Constraints keep their .id attribute, as it is used to propagate dual variables.

        Summary:
        ========
        Leafs:              Same object
        Constraints:        New object with same .id
        Other expressions:  New object with new .id
        """
        return self.copy()

    def __deepcopy__(self, memo):
        """
        Called by copy.deepcopy()
        Creates an independent copy of the object while maintaining the relationship between the
        nodes in the expression tree.
        """
        cvxpy_id = getattr(self, 'id', None)
        if cvxpy_id is not None and cvxpy_id in memo:
            return memo[cvxpy_id]
        else:
            with DefaultDeepCopyContextManager(self):
                new = copy.deepcopy(self, memo)
            if getattr(self, 'id', None) is not None:
                new_id = lu.get_id()
                new.id = new_id
            memo[cvxpy_id] = new
            return new

    def get_data(self) -> None:
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return None

    def atoms(self):
        """Returns all the atoms present in the args.

        Returns
        -------
        list
        """
        return unique_list((atom for arg in self.args for atom in arg.atoms()))