import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple
import torch
class BasePruningMethod(ABC):
    """Abstract base class for creation of new pruning techniques.

    Provides a skeleton for customization requiring the overriding of methods
    such as :meth:`compute_mask` and :meth:`apply`.
    """
    _tensor_name: str

    def __call__(self, module, inputs):
        """Multiply the mask into original tensor and store the result.

        Multiplies the mask (stored in ``module[name + '_mask']``)
        into the original tensor (stored in ``module[name + '_orig']``)
        and stores the result into ``module[name]`` by using :meth:`apply_mask`.

        Args:
            module (nn.Module): module containing the tensor to prune
            inputs: not used.
        """
        setattr(module, self._tensor_name, self.apply_mask(module))

    @abstractmethod
    def compute_mask(self, t, default_mask):
        """Compute and returns a mask for the input tensor ``t``.

        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a random mask to
        apply on top of the ``default_mask`` according to the specific pruning
        method recipe.

        Args:
            t (torch.Tensor): tensor representing the importance scores of the
            parameter to prune.
            default_mask (torch.Tensor): Base mask from previous pruning
            iterations, that need to be respected after the new mask is
            applied. Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``
        """
        pass

    def apply_mask(self, module):
        """Simply handles the multiplication between the parameter being pruned and the generated mask.

        Fetches the mask and the original tensor from the module
        and returns the pruned version of the tensor.

        Args:
            module (nn.Module): module containing the tensor to prune

        Returns:
            pruned_tensor (torch.Tensor): pruned version of the input tensor
        """
        assert self._tensor_name is not None, f'Module {module} has to be pruned'
        mask = getattr(module, self._tensor_name + '_mask')
        orig = getattr(module, self._tensor_name + '_orig')
        pruned_tensor = mask.to(dtype=orig.dtype) * orig
        return pruned_tensor

    @classmethod
    def apply(cls, module, name, *args, importance_scores=None, **kwargs):
        """Add pruning on the fly and reparametrization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            args: arguments passed on to a subclass of
                :class:`BasePruningMethod`
            importance_scores (torch.Tensor): tensor of importance scores (of
                same shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the
                corresponding elements in the parameter being pruned.
                If unspecified or None, the parameter will be used in its place.
            kwargs: keyword arguments passed on to a subclass of a
                :class:`BasePruningMethod`
        """

        def _get_composite_method(cls, module, name, *args, **kwargs):
            old_method = None
            found = 0
            hooks_to_remove = []
            for k, hook in module._forward_pre_hooks.items():
                if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
                    old_method = hook
                    hooks_to_remove.append(k)
                    found += 1
            assert found <= 1, f'Avoid adding multiple pruning hooks to the                same tensor {name} of module {module}. Use a PruningContainer.'
            for k in hooks_to_remove:
                del module._forward_pre_hooks[k]
            method = cls(*args, **kwargs)
            method._tensor_name = name
            if old_method is not None:
                if isinstance(old_method, PruningContainer):
                    old_method.add_pruning_method(method)
                    method = old_method
                elif isinstance(old_method, BasePruningMethod):
                    container = PruningContainer(old_method)
                    container.add_pruning_method(method)
                    method = container
            return method
        method = _get_composite_method(cls, module, name, *args, **kwargs)
        orig = getattr(module, name)
        if importance_scores is not None:
            assert importance_scores.shape == orig.shape, f'importance_scores should have the same shape as parameter                 {name} of {module}'
        else:
            importance_scores = orig
        if not isinstance(method, PruningContainer):
            module.register_parameter(name + '_orig', orig)
            del module._parameters[name]
            default_mask = torch.ones_like(orig)
        else:
            default_mask = getattr(module, name + '_mask').detach().clone(memory_format=torch.contiguous_format)
        try:
            mask = method.compute_mask(importance_scores, default_mask=default_mask)
            module.register_buffer(name + '_mask', mask)
            setattr(module, name, method.apply_mask(module))
            module.register_forward_pre_hook(method)
        except Exception as e:
            if not isinstance(method, PruningContainer):
                orig = getattr(module, name + '_orig')
                module.register_parameter(name, orig)
                del module._parameters[name + '_orig']
            raise e
        return method

    def prune(self, t, default_mask=None, importance_scores=None):
        """Compute and returns a pruned version of input tensor ``t``.

        According to the pruning rule specified in :meth:`compute_mask`.

        Args:
            t (torch.Tensor): tensor to prune (of same dimensions as
                ``default_mask``).
            importance_scores (torch.Tensor): tensor of importance scores (of
                same shape as ``t``) used to compute mask for pruning ``t``.
                The values in this tensor indicate the importance of the
                corresponding elements in the ``t`` that is being pruned.
                If unspecified or None, the tensor ``t`` will be used in its place.
            default_mask (torch.Tensor, optional): mask from previous pruning
                iteration, if any. To be considered when determining what
                portion of the tensor that pruning should act on. If None,
                default to a mask of ones.

        Returns:
            pruned version of tensor ``t``.
        """
        if importance_scores is not None:
            assert importance_scores.shape == t.shape, 'importance_scores should have the same shape as tensor t'
        else:
            importance_scores = t
        default_mask = default_mask if default_mask is not None else torch.ones_like(t)
        return t * self.compute_mask(importance_scores, default_mask=default_mask)

    def remove(self, module):
        """Remove the pruning reparameterization from a module.

        The pruned parameter named ``name`` remains permanently pruned,
        and the parameter named ``name+'_orig'`` is removed from the parameter list.
        Similarly, the buffer named ``name+'_mask'`` is removed from the buffers.

        Note:
            Pruning itself is NOT undone or reversed!
        """
        assert self._tensor_name is not None, f'Module {module} has to be pruned            before pruning can be removed'
        weight = self.apply_mask(module)
        if hasattr(module, self._tensor_name):
            delattr(module, self._tensor_name)
        orig = module._parameters[self._tensor_name + '_orig']
        orig.data = weight.data
        del module._parameters[self._tensor_name + '_orig']
        del module._buffers[self._tensor_name + '_mask']
        setattr(module, self._tensor_name, orig)