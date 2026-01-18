import statemachine
import trafficlightstate
class TrafficLight(trafficlightstate.TrafficLightStateMixin):

    def __init__(self):
        self.initialize_state(trafficlightstate.Red)

    def change(self):
        self._state = self._state.next_state()