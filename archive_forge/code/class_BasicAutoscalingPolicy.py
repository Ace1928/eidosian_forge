import logging
import math
from abc import ABCMeta, abstractmethod
from typing import List, Optional
from ray.serve._private.common import TargetCapacityDirection
from ray.serve._private.constants import CONTROL_LOOP_PERIOD_S, SERVE_LOGGER_NAME
from ray.serve._private.utils import get_capacity_adjusted_num_replicas
from ray.serve.config import AutoscalingConfig
class BasicAutoscalingPolicy(AutoscalingPolicy):
    """The default autoscaling policy based on basic thresholds for scaling.
    There is a minimum threshold for the average queue length in the cluster
    to scale up and a maximum threshold to scale down. Each period, a 'scale
    up' or 'scale down' decision is made. This decision must be made for a
    specified number of periods in a row before the number of replicas is
    actually scaled. See config options for more details.  Assumes
    `get_decision_num_replicas` is called once every CONTROL_LOOP_PERIOD_S
    seconds.
    """

    def __init__(self, config: AutoscalingConfig):
        self.config = config
        self.loop_period_s = CONTROL_LOOP_PERIOD_S
        self.scale_up_consecutive_periods = int(config.upscale_delay_s / self.loop_period_s)
        self.scale_down_consecutive_periods = int(config.downscale_delay_s / self.loop_period_s)
        self.decision_counter = 0

    def get_decision_num_replicas(self, curr_target_num_replicas: int, current_num_ongoing_requests: List[float], current_handle_queued_queries: float, target_capacity: Optional[float]=None, target_capacity_direction: Optional[TargetCapacityDirection]=None) -> int:
        if len(current_num_ongoing_requests) == 0:
            if current_handle_queued_queries > 0:
                return max(math.ceil(1 * self.config.get_upscale_smoothing_factor()), curr_target_num_replicas)
            return curr_target_num_replicas
        decision_num_replicas = curr_target_num_replicas
        desired_num_replicas = calculate_desired_num_replicas(self.config, current_num_ongoing_requests, override_min_replicas=self.get_current_lower_bound(target_capacity, target_capacity_direction), override_max_replicas=get_capacity_adjusted_num_replicas(self.config.max_replicas, target_capacity))
        if desired_num_replicas > curr_target_num_replicas:
            if self.decision_counter < 0:
                self.decision_counter = 0
            self.decision_counter += 1
            if self.decision_counter > self.scale_up_consecutive_periods:
                self.decision_counter = 0
                decision_num_replicas = desired_num_replicas
        elif desired_num_replicas < curr_target_num_replicas:
            if self.decision_counter > 0:
                self.decision_counter = 0
            self.decision_counter -= 1
            if self.decision_counter < -self.scale_down_consecutive_periods:
                self.decision_counter = 0
                decision_num_replicas = desired_num_replicas
        else:
            self.decision_counter = 0
        return decision_num_replicas

    def apply_bounds(self, curr_target_num_replicas: int, target_capacity: Optional[float]=None, target_capacity_direction: Optional[TargetCapacityDirection]=None) -> int:
        """Clips curr_target_num_replicas using the current bounds."""
        upper_bound = get_capacity_adjusted_num_replicas(self.config.max_replicas, target_capacity)
        lower_bound = self.get_current_lower_bound(target_capacity, target_capacity_direction)
        return max(lower_bound, min(upper_bound, curr_target_num_replicas))

    def get_current_lower_bound(self, target_capacity: Optional[float]=None, target_capacity_direction: Optional[TargetCapacityDirection]=None) -> int:
        """Get the autoscaling lower bound, including target_capacity changes.

        The autoscaler uses initial_replicas scaled by target_capacity only
        if the target capacity direction is UP.
        """
        if self.config.initial_replicas is not None and target_capacity_direction == TargetCapacityDirection.UP:
            return get_capacity_adjusted_num_replicas(self.config.initial_replicas, target_capacity)
        else:
            return get_capacity_adjusted_num_replicas(self.config.min_replicas, target_capacity)