from typing import Optional
import cirq
from cirq_web import widget
Initializes a BlochSphere.

        Also initializes it's parent class Widget with the bundle file provided.

        Args:
            sphere_radius: the radius of the bloch sphere in the three.js diagram.
                The default value is 5.
            state_vector: a state vector to pass in to be represented.

        Raises:
            ValueError: If the `sphere_radius` is not positive or the `state_vector` is not
                supplied.
        