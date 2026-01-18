from abc import abstractmethod
from typing import List
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.logical.interfaces.optimizer import Rule
from ray.data._internal.logical.interfaces.physical_plan import PhysicalPlan
class ZeroCopyMapFusionRule(Rule):
    """Base abstract class for all zero-copy map fusion rules.

    A zero-copy map fusion rule is a rule that optimizes the transform_fn chain of
    a fused MapOperator. The optimization is usually done by removing unnecessary
    data conversions.

    This base abstract class defines the common util functions. And subclasses
    should implement the `_optimize` method for the concrete optimization
    strategy.
    """

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        self._traverse(plan.dag)
        return plan

    def _traverse(self, op):
        """Traverse the DAG and apply the optimization to each MapOperator."""
        if isinstance(op, MapOperator):
            map_transformer = op.get_map_transformer()
            transform_fns = map_transformer.get_transform_fns()
            new_transform_fns = self._optimize(transform_fns)
            map_transformer.set_transform_fns(new_transform_fns)
        for input_op in op.input_dependencies:
            self._traverse(input_op)

    @abstractmethod
    def _optimize(self, transform_fns: List[MapTransformFn]) -> List[MapTransformFn]:
        """Optimize the transform_fns chain of a MapOperator.

        Args:
            transform_fns: The old transform_fns chain.
        Returns:
            The optimized transform_fns chain.
        """
        ...