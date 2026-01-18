import threading
import copy
import warnings
import re
import json
from collections import OrderedDict, defaultdict
import numpy as np
from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, np_symbol
from ..symbol import Symbol, load_json
from ..ndarray import NDArray
from .. import name as _name
from .parameter import Parameter, ParameterDict, DeferredInitializationError
from .utils import _indent, _brief_print_list, HookHandle
from .utils import _check_same_symbol_type, _check_all_np_ndarrays
from .. import numpy_extension as _mx_npx
from .. import numpy as _mx_np
from .. util import is_np_array, np_shape, np_array
class HybridBlock(Block):
    """`HybridBlock` supports forwarding with both Symbol and NDArray.

    `HybridBlock` is similar to `Block`, with a few differences::

        import mxnet as mx
        from mxnet.gluon import HybridBlock, nn

        class Model(HybridBlock):
            def __init__(self, **kwargs):
                super(Model, self).__init__(**kwargs)
                # use name_scope to give child Blocks appropriate names.
                with self.name_scope():
                    self.dense0 = nn.Dense(20)
                    self.dense1 = nn.Dense(20)

            def hybrid_forward(self, F, x):
                x = F.relu(self.dense0(x))
                return F.relu(self.dense1(x))

        model = Model()
        model.initialize(ctx=mx.cpu(0))
        model.hybridize()
        model(mx.nd.zeros((10, 10), ctx=mx.cpu(0)))

    Forward computation in :py:class:`HybridBlock` must be static to work with :py:class:`Symbol` s,
    i.e. you cannot call :py:meth:`NDArray.asnumpy`, :py:attr:`NDArray.shape`,
    :py:attr:`NDArray.dtype`, `NDArray` indexing (`x[i]`) etc on tensors.
    Also, you cannot use branching or loop logic that bases on non-constant
    expressions like random numbers or intermediate results, since they change
    the graph structure for each iteration.

    Before activating with :py:meth:`hybridize()`, :py:class:`HybridBlock` works just like normal
    :py:class:`Block`. After activation, :py:class:`HybridBlock` will create a symbolic graph
    representing the forward computation and cache it. On subsequent forwards,
    the cached graph will be used instead of :py:meth:`hybrid_forward`.

    Please see references for detailed tutorial.

    References
    ----------
        `Hybrid - Faster training and easy deployment
        <https://mxnet.io/tutorials/gluon/hybrid.html>`_
    """

    def __init__(self, prefix=None, params=None):
        super(HybridBlock, self).__init__(prefix=prefix, params=params)
        self._cached_graph = ()
        self._cached_op = None
        self._cached_op_args = []
        self._out_format = None
        self._in_format = None
        self._active = False
        self._flags = []
        self._callback = None
        self._monitor_all = False
        self._backend = None
        self._backend_opts = {}

    def __setattr__(self, name, value):
        """Registers parameters."""
        super(HybridBlock, self).__setattr__(name, value)
        if isinstance(value, HybridBlock):
            self._clear_cached_op()

    def _get_graph(self, *args):
        if not self._cached_graph:
            flatten_args, self._in_format = _flatten(args, 'input')
            flatten_inputs = []
            symbol_inputs = []
            cnt = 0
            real_arg_num = sum([ele is not None for ele in flatten_args])
            if real_arg_num == 0:
                raise ValueError('All args are None and we do not support such a case. Received args={}'.format(args))
            for arg in flatten_args:
                if arg is not None:
                    if real_arg_num > 1:
                        arg_sym = symbol.var('data{}'.format(cnt))
                    else:
                        arg_sym = symbol.var('data')
                    if isinstance(arg, _mx_np.ndarray):
                        arg_sym = arg_sym.as_np_ndarray()
                    cnt += 1
                    flatten_inputs.append(arg_sym)
                    symbol_inputs.append(arg_sym)
                else:
                    flatten_inputs.append(None)
            grouped_inputs = _regroup(flatten_inputs, self._in_format)
            params = {i: j.var() for i, j in self._reg_params.items()}
            with self.name_scope():
                out = self.hybrid_forward(symbol, *grouped_inputs, **params)
            out, self._out_format = _flatten(out, 'output')
            self._cached_graph = (symbol_inputs, symbol.Group(out, _check_same_symbol_type(out)))
        return self._cached_graph

    def _build_cache(self, *args):
        data, out = self._get_graph(*args)
        data_names = {data.name: i for i, data in enumerate(data)}
        input_names = out.list_inputs()
        expected_names = set(input_names)
        if len(self._cached_op_args) > 0:
            params = {param_tuple[1].name: param_tuple[1] for param_tuple in self._cached_op_args if isinstance(param_tuple[1], Parameter)}
        else:
            params = self.collect_params()
        param_names = set(params.keys())
        for name in expected_names:
            assert name in param_names or name in data_names, 'Unknown input to HybridBlock: %s' % name
        used_data_names = [i for i in data_names if i in expected_names]
        if len(used_data_names) != len(data_names):
            unused = ', '.join(['%d-th' % i for name, i in data_names.items() if name not in expected_names])
            warnings.warn('The %s input to HybridBlock is not used by any computation. Is this intended?' % unused, stacklevel=4)
        used_param_names = [i for i in param_names if i in expected_names]
        if len(used_param_names) != len(param_names):
            unused = ', '.join(list(param_names - set(used_param_names)))
            warnings.warn('Parameter %s is not used by any computation. Is this intended?' % unused, stacklevel=4)
        args, _ = _flatten(args, 'input')
        try:
            for name in input_names:
                if name in params:
                    params[name].data()
        except DeferredInitializationError:
            self._deferred_infer_shape(*args)
            for name in input_names:
                if name in params:
                    params[name]._finish_deferred_init()
        arg_dict, aux_dict = (dict(), dict())
        if self._backend:
            _, _, ctx_set, _ = _gather_type_ctx_info(list(args))
            ctx = ctx_set.pop() if len(ctx_set) > 0 else None
            input_shapes = dict()
            for name in out.list_arguments():
                if name in data_names.keys() and data_names[name] < len(args):
                    if isinstance(args[data_names[name]], NDArray):
                        arg_dict[name] = args[data_names[name]]
                    elif isinstance(args[data_names[name]], symbol.Symbol) and '__shape__' in args[data_names[name]].list_attr():
                        shape_str = args[data_names[name]].list_attr()['__shape__']
                        input_shapes[name] = tuple(map(int, shape_str.strip('()').split(',')))
                elif name in params:
                    arg_dict[name] = params[name].data()
            for name in out.list_auxiliary_states():
                if name in data_names.keys() and data_names[name] < len(args):
                    if isinstance(args[data_names[name]], NDArray):
                        aux_dict[name] = args[data_names[name]]
                    elif isinstance(args[data_names[name]], symbol.Symbol) and '__shape__' in args[data_names[name]].list_attr():
                        shape_str = args[data_names[name]].list_attr()['__shape__']
                        input_shapes[name] = tuple(map(int, shape_str.strip('()').split(',')))
                elif name in params:
                    aux_dict[name] = params[name].data()
            out = out.optimize_for(self._backend, arg_dict, aux_dict, ctx, input_shapes, **self._backend_opts)
            if _mx_npx.is_np_array():
                out = out.as_np_ndarray()
            self._cached_graph = (data, out)
        input_names = out.list_inputs()
        data_indices = []
        param_indices = []
        self._cached_op_args = []
        for i, name in enumerate(input_names):
            pair = None
            if name in data_names:
                data_indices.append(i)
                pair = (True, data_names[name])
            else:
                param_indices.append(i)
                if name in params:
                    param = params[name]
                else:
                    if name in arg_dict or name:
                        param_data = arg_dict[name]
                    elif name in aux_dict:
                        param_data = aux_dict[name]
                    else:
                        raise RuntimeError('A parameter was added to the graph during optimization but it was not added to the parameter dicts.\nPlease check the backend.')
                    param = Parameter(name, dtype=param_data.dtype)
                    param._load_init(param_data, param_data.context)
                pair = (False, param)
            self._cached_op_args.append(pair)
        flags = [('data_indices', data_indices), ('param_indices', param_indices)] + self._flags
        self._cached_op = ndarray.CachedOp(out, flags)

    def _deferred_infer_shape(self, *args):
        try:
            self.infer_shape(*args)
        except Exception as e:
            error_msg = 'Deferred initialization failed because shape cannot be inferred. {}'.format(e)
            raise ValueError(error_msg)

    def _call_cached_op(self, *args):
        if self._cached_op is None:
            self._build_cache(*args)
        assert self._cached_op, 'Gluon failed to build the cache. This should never happen. Please submit an issue on Github https://github.com/apache/incubator-mxnet.'
        if self._callback:
            self._cached_op._register_op_hook(self._callback, self._monitor_all)
            if len(self._flags) >= 2 and (self._flags[1] or self._flags[0]):
                warnings.warn('register_op_hook is experimental when static_alloc=True / static_shape=True  and may not work correctly')
        args, fmt = _flatten(args, 'input')
        if fmt != self._in_format:
            if len(self._in_format) > len(fmt):
                valid = all([self._in_format[i] == -1 for i in range(len(fmt), len(self._in_format))])
                valid = valid and fmt == self._in_format[:len(fmt)]
            elif len(self._in_format) < len(fmt):
                valid = all([fmt[i] == -1 for i in range(len(self._in_format), len(fmt))])
                valid = valid and fmt[:len(self._in_format)] == self._in_format
            else:
                valid = False
            if not valid:
                raise ValueError('The argument structure of HybridBlock does not match the cached version. Stored format = {}, input format = {}'.format(fmt, self._in_format))
        args_without_none = [ele for ele in args if ele is not None]
        cargs = [args_without_none[i] if is_arg else i.data() for is_arg, i in self._cached_op_args]
        out = self._cached_op(*cargs)
        if isinstance(out, NDArray):
            out = [out]
        return _regroup(out, self._out_format)

    def optimize_for(self, x, *args, backend=None, clear=False, static_alloc=False, static_shape=False, inline_limit=2, forward_bulk_size=None, backward_bulk_size=None, **kwargs):
        """Partitions the current HybridBlock and optimizes it for a given backend
        without executing a forward pass. Modifies the HybridBlock in-place.

        Immediately partitions a HybridBlock using the specified backend. Combines
        the work done in the hybridize API with part of the work done in the forward
        pass without calling the CachedOp. Can be used in place of hybridize,
        afterwards `export` can be called or inference can be run. See README.md in
        example/extensions/lib_subgraph/README.md for more details.

        Examples
        --------
        # partition and then export to file
        block.optimize_for(x, backend='myPart')
        block.export('partitioned')

        # partition and then run inference
        block.optimize_for(x, backend='myPart')
        block(x)

        Parameters
        ----------
        x : NDArray
            first input to model
        *args : NDArray
            other inputs to model
        backend : str
            The name of backend, as registered in `SubgraphBackendRegistry`, default None
        clear : bool, default False
            Clears any previous optimizations
        static_alloc : bool, default False
            Statically allocate memory to improve speed. Memory usage may increase.
        static_shape : bool, default False
            Optimize for invariant input shapes between iterations. Must also
            set static_alloc to True. Change of input shapes is still allowed
            but slower.
        inline_limit : optional int, default 2
            Maximum number of operators that can be inlined.
        forward_bulk_size : optional int, default None
            Segment size of bulk execution during forward pass.
        backward_bulk_size : optional int, default None
            Segment size of bulk execution during forward pass.
        **kwargs: The backend options, optional
            Passed on to `PrePartition` and `PostPartition` functions of `SubgraphProperty`
        """
        if len(kwargs) > 0:
            self._backend_opts = kwargs
        if not backend:
            raise ValueError('Must specify "backend" to optimize_for')
        self.hybridize(True, backend, clear, static_alloc, static_shape, inline_limit, forward_bulk_size, backward_bulk_size)
        has_symbol, has_ndarray, ctx_set, _ = _gather_type_ctx_info([x] + list(args))
        if not has_symbol and (not has_ndarray):
            raise ValueError('In HybridBlock, there must be one NDArray or one Symbol in the input. Please check the type of the args.\n')
        if len(ctx_set) > 1:
            raise ValueError('Found multiple contexts in the input, After hybridized, the HybridBlock only supports one input context. You can print the ele.ctx in the input arguments to inspect their contexts. Find all contexts = {}'.format(ctx_set))
        self._build_cache(x, *args)
        assert self._cached_op, 'Gluon failed to build the cache. This should never happen. Please submit an issue on Github https://github.com/apache/incubator-mxnet.'

    def _clear_cached_op(self):
        self._cached_graph = ()
        self._cached_op = None
        self._cached_op_args = []

    def register_child(self, block, name=None):
        if not isinstance(block, HybridBlock):
            raise ValueError('Children of HybridBlock must also be HybridBlock, but %s has type %s. If you are using Sequential, please try HybridSequential instead.' % (str(block), str(type(block))))
        super(HybridBlock, self).register_child(block, name)
        self._clear_cached_op()

    def hybridize(self, active=True, backend=None, clear=True, static_alloc=False, static_shape=False, inline_limit=2, forward_bulk_size=None, backward_bulk_size=None, **kwargs):
        """Activates or deactivates :py:class:`HybridBlock` s recursively. Has no effect on
        non-hybrid children.

        Parameters
        ----------
        active : bool, default True
            Whether to turn hybrid on or off.
        backend : str
            The name of backend, as registered in `SubgraphBackendRegistry`, default None
        clear : bool, default True
            Clears any previous optimizations
        static_alloc : optional bool, default False
            Statically allocate memory to improve speed. Memory usage may increase.
        static_shape : optional bool, default False
            Optimize for invariant input shapes between iterations. Must also
            set static_alloc to True. Change of input shapes is still allowed
            but slower.
        inline_limit : optional int, default 2
            Maximum number of operators that can be inlined.
        forward_bulk_size : optional int, default None
            Segment size of bulk execution during forward pass.
        backward_bulk_size : optional int, default None
            Segment size of bulk execution during forward pass.
        **kwargs:  optional
            Backend options.
        """
        if len(kwargs) > 0:
            self._backend_opts = kwargs
        self._backend = backend
        self._active = active
        self._flags = [('static_alloc', static_alloc), ('static_shape', static_shape), ('inline_limit', inline_limit)]
        if forward_bulk_size is not None:
            self._flags.append(('forward_bulk_size', forward_bulk_size))
        if backward_bulk_size is not None:
            self._flags.append(('backward_bulk_size', backward_bulk_size))
        if clear:
            self._clear_cached_op()
        if active and self._forward_hooks or self._forward_pre_hooks:
            warnings.warn('"{block}" is being hybridized while still having forward hook/pre-hook. If "{block}" is a child of HybridBlock, the hooks will not take effect.'.format(block=self))
        super(HybridBlock, self).hybridize(active, static_alloc=static_alloc, static_shape=static_shape, inline_limit=inline_limit, forward_bulk_size=forward_bulk_size, backward_bulk_size=backward_bulk_size)

    def cast(self, dtype):
        self._clear_cached_op()
        super(HybridBlock, self).cast(dtype)

    def _infer_attrs(self, infer_fn, attr, *args):
        """Generic infer attributes."""
        inputs, out = self._get_graph(*args)
        args, _ = _flatten(args, 'input')
        args_without_none = [ele for ele in args if ele is not None]
        with warnings.catch_warnings(record=True) as w:
            arg_attrs, _, aux_attrs = getattr(out, infer_fn)(**{i.name: getattr(j, attr) for i, j in zip(inputs, args_without_none)})
            if arg_attrs is None:
                raise ValueError(w[0].message)
        sdict = {i: j for i, j in zip(out.list_arguments(), arg_attrs)}
        sdict.update({name: attr for name, attr in zip(out.list_auxiliary_states(), aux_attrs)})
        for i in self.collect_params().values():
            setattr(i, attr, sdict[i.name])

    def infer_shape(self, *args):
        """Infers shape of Parameters from inputs."""
        self._infer_attrs('infer_shape', 'shape', *args)

    def infer_type(self, *args):
        """Infers data type of Parameters from inputs."""
        self._infer_attrs('infer_type', 'dtype', *args)

    def export(self, path, epoch=0, remove_amp_cast=True):
        """Export HybridBlock to json format that can be loaded by
        `gluon.SymbolBlock.imports`, `mxnet.mod.Module` or the C++ interface.

        .. note:: When there are only one input, it will have name `data`. When there
                  Are more than one inputs, they will be named as `data0`, `data1`, etc.

        Parameters
        ----------
        path : str
            Path to save model. Two files `path-symbol.json` and `path-xxxx.params`
            will be created, where xxxx is the 4 digits epoch number.
        epoch : int
            Epoch number of saved model.
        """
        if not self._cached_graph:
            raise RuntimeError('Please first call block.hybridize() and then run forward with this block at least once before calling export.')
        sym = self._cached_graph[1]
        sym.save('%s-symbol.json' % path, remove_amp_cast=remove_amp_cast)
        arg_names = set(sym.list_arguments())
        aux_names = set(sym.list_auxiliary_states())
        arg_dict = {}
        for is_arg, param in self._cached_op_args:
            if not is_arg:
                name = param.name
                if name in arg_names:
                    arg_dict['arg:{}'.format(name)] = param._reduce()
                elif name not in aux_names:
                    warnings.warn('Parameter "{name}" is not found in the graph. '.format(name=name), stacklevel=3)
                else:
                    arg_dict['aux:%s' % name] = param._reduce()
        save_fn = _mx_npx.save if is_np_array() else ndarray.save
        save_fn('%s-%04d.params' % (path, epoch), arg_dict)

    def register_op_hook(self, callback, monitor_all=False):
        """Install op hook for block recursively.

        Parameters
        ----------
        callback : function
            Takes a string and a NDArrayHandle.
        monitor_all : bool, default False
            If true, monitor both input and output, otherwise monitor output only.
        """
        self._callback = callback
        self._monitor_all = monitor_all
        for cld in self._children.values():
            cld._callback = callback
            cld._monitor_all = monitor_all

    def forward(self, x, *args):
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        has_symbol, has_ndarray, ctx_set, first_ctx = _gather_type_ctx_info([x] + list(args))
        if has_symbol and has_ndarray:
            raise ValueError('In HybridBlock, we do not support mixed NDArrays and Symbols types for the input. Please check the type of the args.\n')
        if not has_symbol and (not has_ndarray):
            raise ValueError('In HybridBlock, there must be one NDArray or one Symbol in the input. Please check the type of the args.\n')
        if has_ndarray:
            ctx = first_ctx
            if self._active:
                if len(ctx_set) > 1:
                    raise ValueError('Find multiple contexts in the input, After hybridized, the HybridBlock only supports one input context. You can print the ele.ctx in the input arguments to inspect their contexts. Find all contexts = {}'.format(ctx_set))
                with ctx:
                    return self._call_cached_op(x, *args)
            with ctx:
                try:
                    params = {k: v.data(ctx) for k, v in self._reg_params.items()}
                except DeferredInitializationError:
                    self._deferred_infer_shape(x, *args)
                    for _, v in self.params.items():
                        v._finish_deferred_init()
                    params = {k: v.data(ctx) for k, v in self._reg_params.items()}
                return self.hybrid_forward(ndarray, x, *args, **params)
        params = {i: j.var() for i, j in self._reg_params.items()}
        with self.name_scope():
            return self.hybrid_forward(symbol, x, *args, **params)

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Overrides to construct symbolic graph for this `Block`.

        Parameters
        ----------
        x : Symbol or NDArray
            The first input tensor.
        *args : list of Symbol or list of NDArray
            Additional input tensors.
        """
        raise NotImplementedError