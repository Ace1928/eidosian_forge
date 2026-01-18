import numpy as np
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.gaussian_process.kernels import Kernel
import inspect
def TPOTOperatorClassFactory(opsourse, opdict, BaseClass=Operator, ArgBaseClass=ARGType, verbose=0):
    """Dynamically create operator class.

    Parameters
    ----------
    opsourse: string
        operator source in config dictionary (key)
    opdict: dictionary
        operator params in config dictionary (value)
    regression: bool
        True if it can be used in TPOTRegressor
    classification: bool
        True if it can be used in TPOTClassifier
    BaseClass: Class
        inherited BaseClass for operator
    ArgBaseClass: Class
        inherited BaseClass for parameter
    verbose: int, optional (default: 0)
        How much information TPOT communicates while it's running.
        0 = none, 1 = minimal, 2 = high, 3 = all.
        if verbose > 2 then ImportError will rasie during initialization

    Returns
    -------
    op_class: Class
        a new class for a operator
    arg_types: list
        a list of parameter class

    """
    class_profile = {}
    dep_op_list = {}
    dep_op_type = {}
    import_str, op_str, op_obj = source_decode(opsourse, verbose=verbose)
    if not op_obj:
        return (None, None)
    else:
        if is_classifier(op_obj):
            class_profile['root'] = True
            optype = 'Classifier'
        elif is_regressor(op_obj):
            class_profile['root'] = True
            optype = 'Regressor'
        elif _is_selector(op_obj):
            optype = 'Selector'
        elif _is_transformer(op_obj):
            optype = 'Transformer'
        elif _is_resampler(op_obj):
            optype = 'Resampler'
        else:
            raise ValueError('optype must be one of: Classifier, Regressor, Selector, Transformer, or Resampler')

        @classmethod
        def op_type(cls):
            """Return the operator type.

            Possible values:
                "Classifier", "Regressor", "Selector", "Transformer"
            """
            return optype
        class_profile['type'] = op_type
        class_profile['sklearn_class'] = op_obj
        import_hash = {}
        import_hash[import_str] = [op_str]
        arg_types = []
        for pname in sorted(opdict.keys()):
            prange = opdict[pname]
            if not isinstance(prange, dict):
                classname = '{}__{}'.format(op_str, pname)
                arg_types.append(ARGTypeClassFactory(classname, prange, ArgBaseClass))
            else:
                for dkey, dval in prange.items():
                    dep_import_str, dep_op_str, dep_op_obj = source_decode(dkey, verbose=verbose)
                    if dep_import_str in import_hash:
                        import_hash[dep_import_str].append(dep_op_str)
                    else:
                        import_hash[dep_import_str] = [dep_op_str]
                    dep_op_list[pname] = dep_op_str
                    dep_op_type[pname] = dep_op_obj
                    if dval:
                        for dpname in sorted(dval.keys()):
                            dprange = dval[dpname]
                            classname = '{}__{}__{}'.format(op_str, dep_op_str, dpname)
                            arg_types.append(ARGTypeClassFactory(classname, dprange, ArgBaseClass))
        class_profile['arg_types'] = tuple(arg_types)
        class_profile['import_hash'] = import_hash
        class_profile['dep_op_list'] = dep_op_list
        class_profile['dep_op_type'] = dep_op_type

        @classmethod
        def parameter_types(cls):
            """Return the argument and return types of an operator.

            Parameters
            ----------
            None

            Returns
            -------
            parameter_types: tuple
                Tuple of the DEAP parameter types and the DEAP return type for the
                operator

            """
            return ([np.ndarray] + arg_types, np.ndarray)
        class_profile['parameter_types'] = parameter_types

        @classmethod
        def export(cls, *args):
            """Represent the operator as a string so that it can be exported to a file.

            Parameters
            ----------
            args
                Arbitrary arguments to be passed to the operator

            Returns
            -------
            export_string: str
                String representation of the sklearn class with its parameters in
                the format:
                SklearnClassName(param1="val1", param2=val2)

            """
            op_arguments = []
            if dep_op_list:
                dep_op_arguments = {}
                for dep_op_str in dep_op_list.values():
                    dep_op_arguments[dep_op_str] = []
            for arg_class, arg_value in zip(arg_types, args):
                aname_split = arg_class.__name__.split('__')
                if isinstance(arg_value, str):
                    arg_value = '"{}"'.format(arg_value)
                if len(aname_split) == 2:
                    op_arguments.append('{}={}'.format(aname_split[-1], arg_value))
                else:
                    dep_op_arguments[aname_split[1]].append('{}={}'.format(aname_split[-1], arg_value))
            tmp_op_args = []
            if dep_op_list:
                for dep_op_pname, dep_op_str in dep_op_list.items():
                    arg_value = dep_op_str
                    doptype = dep_op_type[dep_op_pname]
                    if inspect.isclass(doptype):
                        if issubclass(doptype, BaseEstimator) or is_classifier(doptype) or is_regressor(doptype) or _is_transformer(doptype) or _is_resampler(doptype) or issubclass(doptype, Kernel):
                            arg_value = '{}({})'.format(dep_op_str, ', '.join(dep_op_arguments[dep_op_str]))
                    tmp_op_args.append('{}={}'.format(dep_op_pname, arg_value))
            op_arguments = tmp_op_args + op_arguments
            return '{}({})'.format(op_obj.__name__, ', '.join(op_arguments))
        class_profile['export'] = export
        op_classname = 'TPOT_{}'.format(op_str)
        op_class = type(op_classname, (BaseClass,), class_profile)
        op_class.__name__ = op_str
        return (op_class, arg_types)