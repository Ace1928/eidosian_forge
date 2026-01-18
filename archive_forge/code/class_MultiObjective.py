from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import metrics_tracking
class MultiObjective(Objective):
    """A container for a list of objectives.

    Args:
        objectives: A list of `Objective`s.
    """

    def __init__(self, objectives):
        super().__init__(name='multi_objective', direction='min')
        self.objectives = objectives
        self.name_to_direction = {objective.name: objective.direction for objective in self.objectives}

    def has_value(self, logs):
        return all((key in logs for key in self.name_to_direction))

    def get_value(self, logs):
        obj_value = 0
        for metric_name, metric_value in logs.items():
            if metric_name not in self.name_to_direction:
                continue
            if self.name_to_direction[metric_name] == 'min':
                obj_value += metric_value
            else:
                obj_value -= metric_value
        return obj_value

    def __eq__(self, obj):
        if self.name_to_direction.keys() != obj.name_to_direction.keys():
            return False
        return sorted(self.objectives, key=lambda x: x.name) == sorted(obj.objectives, key=lambda x: x.name)

    def __str__(self):
        return 'Multi' + super().__str__() + f': [{', '.join(map(lambda x: str(x), self.objectives))}]'