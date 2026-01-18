import unittest
from unittest.mock import patch, MagicMock
from FractalNeuralNetwork import main


class TestFractalNeuralNetwork(unittest.TestCase):
    @patch("FractalNeuralNetwork.create_mini_network")
    @patch("FractalNeuralNetwork.dynamic_input_generator")
    @patch("FractalNeuralNetwork.SimulationParameters")
    def test_main(
        self, MockSimulationParameters, MockDynamicInputGenerator, MockCreateMiniNetwork
    ):
        # Arrange
        mock_simulation_params = MockSimulationParameters()
        mock_mini_network = MockCreateMiniNetwork()
        mock_input_signals = MockDynamicInputGenerator()
        mock_mini_network.simulate.return_value = "mock_output"
        time_steps = 100

        # Act
        try:
            main(time_steps)
        except Exception as e:
            self.fail(f"main() raised Exception unexpectedly: {e}")

        # Assert
        MockSimulationParameters.assert_called_once()
        MockCreateMiniNetwork.assert_called_once_with(mock_simulation_params)
        MockDynamicInputGenerator.assert_called_once_with(
            time_steps, mock_simulation_params
        )
        mock_mini_network.simulate.assert_called_with(
            mock_input_signals[time_steps - 1], time_steps - 1, mock_simulation_params
        )

    @patch("FractalNeuralNetwork.create_mini_network")
    @patch("FractalNeuralNetwork.dynamic_input_generator")
    @patch("FractalNeuralNetwork.SimulationParameters")
    def test_main_exception(
        self, MockSimulationParameters, MockDynamicInputGenerator, MockCreateMiniNetwork
    ):
        # Arrange
        mock_simulation_params = MockSimulationParameters()
        mock_mini_network = MockCreateMiniNetwork()
        mock_input_signals = MockDynamicInputGenerator()
        mock_mini_network.simulate.side_effect = Exception("Test Exception")
        time_steps = 100

        # Act & Assert
        with self.assertRaises(Exception) as context:
            main(time_steps)
        self.assertTrue(
            "An error occurred during the simulation: Test Exception"
            in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
