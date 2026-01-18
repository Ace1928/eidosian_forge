import numpy as np
class TreeEnsembleAttributes:

    def __init__(self):
        self._names = []

    def add(self, name, value):
        if not name.endswith('_as_tensor'):
            self._names.append(name)
        if isinstance(value, list):
            if name in {'base_values', 'class_weights', 'nodes_values', 'nodes_hitrates'}:
                value = np.array(value, dtype=np.float32)
            elif name.endswith('as_tensor'):
                value = np.array(value)
        setattr(self, name, value)

    def __str__(self):
        rows = ['Attributes']
        for name in self._names:
            if name.endswith('_as_tensor'):
                name = name.replace('_as_tensor', '')
            rows.append(f'  {name}={getattr(self, name)}')
        return '\n'.join(rows)