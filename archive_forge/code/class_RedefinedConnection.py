from lognormal_around import lognormal_around
import random
class RedefinedConnection:
    """
    Redefined Connection class with modified Hebbian learning based on threshold crossings.
    """

    def __init__(self, global_scaling_factor=1.0, is_excitatory=True):
        self.strength = lognormal_around(2.75, 0.5, 5) * global_scaling_factor
        self.delay = int(lognormal_around(3, 1, 5))
        self.is_excitatory = is_excitatory
        self.global_scaling_factor = global_scaling_factor
        self.excitotoxic_inversion_steps = 0

    def update_strength(self, pre_activity, post_activity, post_neuron):
        """
        Update the connection strength based on neuron activity, considering threshold crossings.
        """
        learning_rate = 0.01 * self.global_scaling_factor
        strength_change = pre_activity * post_activity * learning_rate
        if post_neuron.output >= post_neuron.over_threshold:
            strength_change *= -1
        if post_neuron.output >= post_neuron.excitotoxic_threshold:
            self.excitotoxic_inversion_steps = random.randint(20, 40)
        if self.excitotoxic_inversion_steps > 0:
            strength_change *= -1
            self.excitotoxic_inversion_steps -= 1
        self.strength += strength_change
        self.strength = max(0, self.strength)

    def get_effective_strength(self):
        """
        Get the effective strength of the connection, considering if it's excitatory or inhibitory.
        """
        if self.is_excitatory:
            return self.strength
        else:
            return -self.strength