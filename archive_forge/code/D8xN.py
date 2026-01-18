# Standard Python libraries
import logging
import random
from typing import Optional, List, Tuple

# For GUI elements
from tkinter import filedialog, messagebox
import tkinter as tk

# K -Arnold Network (KAN) class from the pykan library (imported as kan)
from kan import KAN

# PyTorch library for tensor operations
import torch

# For multi-threading in the GUI application
from threading import Thread

# For plotting functions
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class KANWrapper:
    def __init__(
        self,
        width: List[int],
        grid: int,
        k: int,
        learnable_params: Optional[dict] = None,
        lr: float = 0.01,
    ) -> None:
        """
        Initialize the Kolmogorov-Arnold Network (KAN).

        Args:
            width (List[int]): List of integers specifying the width of each layer.
            grid (int): Grid size for the KAN.
            k (int): Number of neurons in the hidden layer.
            learnable_params (Optional[dict]): Optional dictionary of additional learnable parameters.
            lr (float): Learning rate for the optimizer.
        """
        self.width = width
        self.grid = grid
        self.k = k
        self.learnable_params = learnable_params or {}
        self.model = KAN(width=width, grid=grid, k=k)
        self._initialize_parameters()
        self._initialize_missing_keys()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learnable_params.get("lr", lr)
        )
        logging.info(
            "Initialized KANWrapper with width: %s, grid: %d, k: %d", width, grid, k
        )

    def _initialize_parameters(self) -> None:
        """
        Initialize all model parameters and learnable parameters with log-normal distribution.
        """
        for param in self.model.parameters():
            self._initialize_tensor(param)

        for key, value in self.learnable_params.items():
            self.learnable_params[key] = self._initialize_tensor(
                value if isinstance(value, torch.Tensor) else torch.tensor(value)
            )

    @staticmethod
    def _initialize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """
        Initialize a tensor with log-normal distribution.

        Args:
            tensor (torch.Tensor): The tensor to initialize.

        Returns:
            torch.Tensor: The initialized tensor.
        """
        return torch.randn_like(tensor).log_normal_()

    def _initialize_missing_keys(self) -> None:
        """
        Ensure all required keys are present in learnable_params and initialize them if missing.
        """
        required_keys = self._get_required_keys()

        for key in required_keys:
            if key not in self.learnable_params:
                self.learnable_params[key] = torch.randn(1).log_normal_()
                logging.debug(
                    "Initialized missing key %s with log-normal distribution", key
                )

    def _get_required_keys(self) -> List[str]:
        """
        Get the list of required keys for learnable parameters dynamically from the model.

        Returns:
            List[str]: List of required keys.
        """
        return [name for name, _ in self.model.named_parameters()]

    def save(self, filename: str) -> None:
        """
        Save the network's weights, biases, and learnable parameters to a file.

        Args:
            filename (str): The file path to save the network.
        """
        state = {
            "model_state_dict": self.model.state_dict(),
            "width": self.width,
            "grid": self.grid,
            "k": self.k,
            "learnable_params": self.learnable_params,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        with open(filename, "wb") as f:
            torch.save(state, f)
        logging.info("Saved KANWrapper model to %s", filename)

    @classmethod
    def load(cls, filename: str) -> "KANWrapper":
        """
        Load a network's weights, biases, and learnable parameters from a file.

        Args:
            filename (str): The file path to load the network from.

        Returns:
            KANWrapper: An instance of the KANWrapper class with loaded weights, biases, and learnable parameters.
        """
        with open(filename, "rb") as f:
            state = torch.load(f)
        missing_keys = []
        required_keys = [
            "width",
            "grid",
            "k",
            "model_state_dict",
            "learnable_params",
            "optimizer_state_dict",
        ]

        # Initialize missing keys with log-normal distribution
        for key in required_keys:
            if key not in state:
                if key == "learnable_params":
                    state[key] = {}
                else:
                    state[key] = torch.randn(1).log_normal_()
                missing_keys.append(key)

        if missing_keys:
            logging.warning(
                "Missing keys in loaded state: %s. Filled with default values.",
                missing_keys,
            )
            if messagebox.askyesno(
                "Missing Keys",
                f"Missing keys: {missing_keys}. Do you want to save the corrected model?",
            ):
                with open(filename, "wb") as f:
                    torch.save(state, f)
                logging.info("Saved corrected model to %s", filename)

        instance = cls(
            width=state["width"],
            grid=state["grid"],
            k=state["k"],
            learnable_params=state["learnable_params"],
        )
        instance.model.load_state_dict(state["model_state_dict"])
        if state["optimizer_state_dict"]:
            instance.optimizer = torch.optim.Adam(instance.model.parameters())
            instance.optimizer.load_state_dict(state["optimizer_state_dict"])
        else:
            instance.optimizer = torch.optim.Adam(
                instance.model.parameters(),
                lr=state["learnable_params"].get("lr", 0.01),
            )
        logging.info("Loaded KANWrapper model from %s", filename)
        return instance

    def initialize_from_another_model(
        self, other: "KANWrapper", x: torch.Tensor
    ) -> None:
        """
        Initialize the model from another parent model.

        Args:
            other (KANWrapper): The parent model to initialize from.
            x (torch.Tensor): Input tensor for initialization.
        """
        self.model.initialize_from_another_model(other.model, x)
        logging.info("Initialized KANWrapper from another model")

    def fix_symbolic(
        self,
        layer: int,
        neuron: int,
        grid: int,
        function: str,
        fit_params_bool: bool = False,
    ) -> None:
        """
        Fix a symbolic function to the activations.

        Args:
            layer (int): The layer index.
            neuron (int): The neuron index.
            grid (int): The grid index.
            function (str): The symbolic function to fix.
            fit_params_bool (bool): Whether to fit parameters.
        """
        self.model.fix_symbolic(layer, neuron, grid, function, fit_params_bool)
        logging.info(
            "Fixed symbolic function %s to layer %d, neuron %d, grid %d",
            function,
            layer,
            neuron,
            grid,
        )

    def get_range(self, layer: int, neuron: int, grid: int) -> Tuple[float, float]:
        """
        Get the input and output range of a specific activation.

        Args:
            layer (int): The layer index.
            neuron (int): The neuron index.
            grid (int): The grid index.

        Returns:
            Tuple[float, float]: The input and output range.
        """
        range_values = self.model.get_range(layer, neuron, grid)
        logging.info(
            "Got range for layer %d, neuron %d, grid %d: %s",
            layer,
            neuron,
            grid,
            range_values,
        )
        return range_values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        return self.model(x)

    def train(
        self,
        dataset: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 0.01,
        epochs: int = 100,
    ) -> None:
        """
        Train the network using a simple gradient descent algorithm.

        Args:
            dataset (torch.Tensor): Input data for training.
            targets (torch.Tensor): Target outputs for training.
            learning_rate (float): Learning rate for gradient descent.
            epochs (int): Number of training epochs.

        Raises:
            ValueError: If the model parameters are empty.
        """
        if not list(self.model.parameters()):
            raise ValueError(
                "Model parameters are empty. Ensure the model is initialized correctly."
            )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(dataset)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            logging.debug("Epoch [%d/%d], Loss: %.4f", epoch + 1, epochs, loss.item())

    def mutate(self, mutation_rate: float = 0.1) -> None:
        """
        Mutate the network's weights, biases, and learnable parameters.

        Args:
            mutation_rate (float): Probability of mutation for each weight, bias, and learnable parameter.
        """
        for param in self.model.parameters():
            if random.random() < mutation_rate:
                param.data.add_(torch.randn_like(param) * mutation_rate)
        for key, value in self.learnable_params.items():
            if random.random() < mutation_rate:
                self.learnable_params[key].add_(torch.randn_like(value) * mutation_rate)
        logging.info("Mutated KANWrapper with mutation rate: %f", mutation_rate)

    def inherit(self, other: "KANWrapper") -> None:
        """
        Inherit weights, biases, and learnable parameters from another KAN instance.

        Args:
            other (KANWrapper): Another KAN instance to inherit from.
        """
        for param_self, param_other in zip(
            self.model.parameters(), other.model.parameters()
        ):
            param_self.data = (param_self.data + param_other.data) / 2
        for key in self.learnable_params:
            if key in other.learnable_params:
                self.learnable_params[key] = (
                    self.learnable_params[key] + other.learnable_params[key]
                ) / 2
        logging.info(
            "Inherited weights, biases, and learnable parameters from another KANWrapper"
        )

    def plot_results(
        self, x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor
    ) -> None:
        """
        Plot the results of the model's predictions.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): True target values.
            y_pred (torch.Tensor): Predicted values from the model.
        """
        plt.figure(figsize=(10, 5))
        plt.scatter(x[:, 0].numpy(), y[:, 0].numpy(), label="True Values")
        plt.scatter(
            x[:, 0].numpy(), y_pred[:, 0].detach().numpy(), label="Predicted Values"
        )
        plt.legend()
        plt.show()
        logging.info("Plotted results")


class KANGUI:
    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the KANGUI application.

        Args:
            root (tk.Tk): The root Tkinter window.
        """
        self.root = root
        self.root.title("KAN Demonstration and Example App")
        self.kan_instance: Optional[KANWrapper] = None

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self) -> None:
        """
        Create and pack the GUI widgets.
        """
        self.param_frame = self._create_param_frame()
        self.train_frame = self._create_train_frame()
        self.file_frame = self._create_file_frame()
        self.plot_frame = self._create_plot_frame()
        self.additional_frame = self._create_additional_frame()
        self.default_frame = self._create_default_frame()

    def _create_param_frame(self) -> tk.Frame:
        param_frame = tk.Frame(self.root)
        param_frame.pack(pady=10)

        tk.Label(param_frame, text="Width:").grid(row=0, column=0)
        self.width_entry = tk.Entry(param_frame)
        self.width_entry.grid(row=0, column=1)

        tk.Label(param_frame, text="Grid:").grid(row=1, column=0)
        self.grid_entry = tk.Entry(param_frame)
        self.grid_entry.grid(row=1, column=1)

        tk.Label(param_frame, text="Neurons (k):").grid(row=2, column=0)
        self.k_entry = tk.Entry(param_frame)
        self.k_entry.grid(row=2, column=1)

        tk.Button(
            param_frame, text="Initialize Model", command=self.initialize_model
        ).grid(row=3, columnspan=2, pady=10)

        return param_frame

    def _create_train_frame(self) -> tk.Frame:
        train_frame = tk.Frame(self.root)
        train_frame.pack(pady=10)

        tk.Label(train_frame, text="Learning Rate:").grid(row=0, column=0)
        self.lr_entry = tk.Entry(train_frame)
        self.lr_entry.grid(row=0, column=1)

        tk.Label(train_frame, text="Epochs:").grid(row=1, column=0)
        self.epochs_entry = tk.Entry(train_frame)
        self.epochs_entry.grid(row=1, column=1)

        tk.Button(train_frame, text="Train Model", command=self.train_model).grid(
            row=2, columnspan=2, pady=10
        )

        return train_frame

    def _create_file_frame(self) -> tk.Frame:
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=10)

        tk.Button(file_frame, text="Save Model", command=self.save_model).grid(
            row=0, column=0, padx=5
        )
        tk.Button(file_frame, text="Load Model", command=self.load_model).grid(
            row=0, column=1, padx=5
        )

        return file_frame

    def _create_plot_frame(self) -> tk.Frame:
        plot_frame = tk.Frame(self.root)
        plot_frame.pack(pady=10)

        tk.Button(plot_frame, text="Plot Results", command=self.plot_results).pack()

        return plot_frame

    def _create_additional_frame(self) -> tk.Frame:
        additional_frame = tk.Frame(self.root)
        additional_frame.pack(pady=10)

        tk.Button(
            additional_frame, text="Mutate Model", command=self.mutate_model
        ).grid(row=0, column=0, padx=5)
        tk.Button(
            additional_frame, text="Inherit Model", command=self.inherit_model
        ).grid(row=0, column=1, padx=5)
        tk.Button(
            additional_frame,
            text="Initialize from Another Model",
            command=self.initialize_from_another_model,
        ).grid(row=1, columnspan=2, pady=10)

        return additional_frame

    def _create_default_frame(self) -> tk.Frame:
        default_frame = tk.Frame(self.root)
        default_frame.pack(pady=10)

        tk.Button(
            default_frame,
            text="Proceed with Defaults",
            command=self.proceed_with_defaults,
        ).pack()

        return default_frame

    def initialize_model(self) -> None:
        """
        Initialize the KAN model with the provided parameters.

        Raises:
            ValueError: If there is an error parsing the inputs.
            RuntimeError: If there is a critical error initializing the model.
        """
        try:
            width, grid, k = self._parse_inputs()
            self._init_model(width, grid, k)
        except ValueError as e:
            logging.error("Error in initialize_model: %s", e)
            self._handle_missing_params()

    def _parse_inputs(self) -> Tuple[List[int], int, int]:
        """
        Parse the inputs from the GUI entries.

        Returns:
            Tuple[List[int], int, int]: Parsed width, grid, and k values.

        Raises:
            ValueError: If there is an error parsing the inputs.
        """
        try:
            width = list(map(int, self.width_entry.get().split(",")))
            grid = int(self.grid_entry.get())
            k = int(self.k_entry.get())
            return width, grid, k
        except ValueError as e:
            logging.error("ValueError parsing inputs: %s", e)
            messagebox.showerror("Error", f"ValueError parsing inputs: {e}")
            raise

    def _init_model(self, width: List[int], grid: int, k: int) -> None:
        """
        Initialize the KAN model with the provided parameters.

        Args:
            width (List[int]): Width of the model.
            grid (int): Grid size.
            k (int): Number of neurons.

        Raises:
            RuntimeError: If there is a critical error initializing the model.
        """
        try:
            self.kan_instance = KANWrapper(width=width, grid=grid, k=k)
            messagebox.showinfo("Success", "Model initialized successfully!")
        except RuntimeError as e:
            logging.critical("Critical error initializing model: %s", e)
            messagebox.showerror("Error", f"Critical error initializing model: %s", e)
            raise

    def _handle_missing_params(self) -> None:
        """
        Handle missing parameters by generating defaults and reinitializing the model.
        """
        width = list(map(int, torch.exp(torch.normal(0, 1, size=(3,))).tolist()))
        grid = int(torch.exp(torch.normal(0, 1)).item())
        k = int(torch.exp(torch.normal(0, 1)).item())
        logging.info(
            "Generated default parameters: width=%s, grid=%d, k=%d", width, grid, k
        )
        self._init_model(width, grid, k)
        if messagebox.askyesno(
            "Confirm",
            "Parameters were missing or invalid. Defaults have been generated. Do you want to save the model with these parameters?",
        ):
            self.save_model()

    def train_model(self) -> None:
        """
        Train the KAN model using the provided parameters.

        Raises:
            ValueError: If the model is not initialized.
            RuntimeError: If there is a critical error during training.
        """
        Thread(target=self._run_training).start()

    def _run_training(self) -> None:
        """
        Run the training process in a separate thread.
        """
        try:
            if self.kan_instance is None:
                raise ValueError("Model is not initialized")

            learning_rate = float(self.lr_entry.get())
            epochs = int(self.epochs_entry.get())

            # Generate some example data
            x = torch.normal(0, 1, size=(100, 2))
            y = x[:, 0] ** 2 + x[:, 1] ** 2  # Example quadratic function
            y = y.unsqueeze(1)

            self.kan_instance.train(x, y, learning_rate=learning_rate, epochs=epochs)
            messagebox.showinfo("Success", "Model trained successfully!")
        except ValueError as e:
            logging.error("ValueError training model: %s", e)
            messagebox.showerror("Error", f"ValueError training model: {e}")
        except RuntimeError as e:
            logging.critical("Critical error training model: %s", e)
            messagebox.showerror("Error", f"Critical error training model: %s", e)

    def proceed_with_defaults(self) -> None:
        """
        Generate a default model, create synthetic data, run training, and offer to save the model.
        """
        try:
            # Generate default parameters
            self._handle_missing_params()  # This will also initialize the model with defaults

            # Generate synthetic data
            x = torch.normal(0, 1, size=(100, 2))
            y = x[:, 0] ** 2 + x[:, 1] ** 2  # Example quadratic function
            y = y.unsqueeze(1)

            # Set default training parameters
            learning_rate = 0.01
            epochs = 50

            # Train the model
            self.kan_instance.train(x, y, learning_rate=learning_rate, epochs=epochs)
            messagebox.showinfo("Success", "Default model trained successfully!")

            # Offer to save the model
            if messagebox.askyesno(
                "Save Model", "Do you want to save the trained model?"
            ):
                self.save_model()
        except Exception as e:
            logging.error("Error during default processing: %s", e)
            messagebox.showerror("Error", f"An error occurred: %s", e)

    def save_model(self) -> None:
        """
        Save the KAN model to a file.

        Raises:
            ValueError: If the model is not initialized.
            RuntimeError: If there is a critical error saving the model.
        """
        try:
            if self.kan_instance is None:
                raise ValueError("Model is not initialized")

            filename = filedialog.asksaveasfilename(
                defaultextension=".pth", filetypes=[("PyTorch Model", "*.pth")]
            )
            if filename:
                self._save_model_to_file(filename)
        except ValueError as e:
            logging.error("ValueError saving model: %s", e)
            messagebox.showerror("Error", f"ValueError saving model: %s", e)
        except RuntimeError as e:
            logging.critical("Critical error saving model: %s", e)
            messagebox.showerror("Error", f"Critical error saving model: %s", e)

    def _save_model_to_file(self, filename: str) -> None:
        """
        Save the KAN model to a file.

        Args:
            filename (str): The file path to save the model.
        """
        self.kan_instance.save(filename)
        messagebox.showinfo("Success", "Model saved successfully!")

    def load_model(self) -> None:
        """
        Load a KAN model from a file.

        Raises:
            ValueError: If there is a value error during loading.
            RuntimeError: If there is a critical error loading the model.
        """
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("PyTorch Model", "*.pth")]
            )
            if filename:
                self._load_model_from_file(filename)
        except ValueError as e:
            logging.error("ValueError loading model: %s", e)
            messagebox.showerror("Error", f"ValueError loading model: %s", e)
        except RuntimeError as e:
            logging.critical("Critical error loading model: %s", e)
            messagebox.showerror("Error", f"Critical error loading model: %s", e)

    def _load_model_from_file(self, filename: str) -> None:
        """
        Load a KAN model from a file.

        Args:
            filename (str): The file path to load the model from.
        """
        self.kan_instance = KANWrapper.load(filename)
        messagebox.showinfo("Success", "Model loaded successfully!")

    def plot_results(self) -> None:
        """
        Plot the results of the model's predictions.

        Raises:
            ValueError: If the model is not initialized.
            RuntimeError: If there is a critical error during plotting.
        """
        Thread(target=self._run_plotting).start()

    def _run_plotting(self) -> None:
        """
        Run the plotting process in a separate thread.
        """
        try:
            if self.kan_instance is None:
                raise ValueError("Model is not initialized")

            # Generate some example data
            x = torch.normal(0, 1, size=(100, 2))
            y = x[:, 0] ** 2 + x[:, 1] ** 2  # Example quadratic function
            y = y.unsqueeze(1)

            y_pred = self.kan_instance.forward(x)

            fig = plt.figure(figsize=(10, 5))
            plt.scatter(x[:, 0].numpy(), y[:, 0].numpy(), label="True Values")
            plt.scatter(
                x[:, 0].numpy(),
                y_pred[:, 0].detach().numpy(),
                label="Predicted Values",
            )
            plt.legend()

            canvas = FigureCanvasTkAgg(fig, master=self.root)
            canvas.draw()
            canvas.get_tk_widget().pack()
            logging.info("Plotted results")
        except ValueError as e:
            logging.error("ValueError plotting results: %s", e)
            messagebox.showerror("Error", f"ValueError plotting results: {e}")
        except RuntimeError as e:
            logging.critical("Critical error plotting results: %s", e)
            messagebox.showerror("Error", f"Critical error plotting results: %s", e)

    def mutate_model(self) -> None:
        """
        Mutate the KAN model with the provided mutation rate.

        Raises:
            ValueError: If the model is not initialized.
            RuntimeError: If there is a critical error during mutation.
        """
        try:
            if self.kan_instance is None:
                raise ValueError("Model is not initialized")

            mutation_rate = float(self.lr_entry.get())
            self.kan_instance.mutate(mutation_rate=mutation_rate)
            messagebox.showinfo("Success", "Model mutated successfully!")
        except ValueError as e:
            logging.error("ValueError mutating model: %s", e)
            messagebox.showerror("Error", f"ValueError mutating model: %s", e)
        except RuntimeError as e:
            logging.critical("Critical error mutating model: %s", e)
            messagebox.showerror("Error", f"Critical error mutating model: %s", e)

    def inherit_model(self) -> None:
        """
        Inherit parameters from another KAN model.

        Raises:
            ValueError: If the model is not initialized.
            RuntimeError: If there is a critical error during inheritance.
        """
        try:
            if self.kan_instance is None:
                raise ValueError("Model is not initialized")

            filename = filedialog.askopenfilename(
                filetypes=[("PyTorch Model", "*.pth")]
            )
            if filename:
                self._inherit_model_from_file(filename)
        except ValueError as e:
            logging.error("ValueError inheriting model: %s", e)
            messagebox.showerror("Error", f"ValueError inheriting model: %s", e)
        except RuntimeError as e:
            logging.critical("Critical error inheriting model: %s", e)
            messagebox.showerror("Error", f"Critical error inheriting model: %s", e)

    def _inherit_model_from_file(self, filename: str) -> None:
        """
        Inherit parameters from another KAN model.

        Args:
            filename (str): The file path to load the model from.
        """
        other_kan_instance = KANWrapper.load(filename)
        self.kan_instance.inherit(other_kan_instance)
        messagebox.showinfo("Success", "Model inherited successfully!")

    def initialize_from_another_model(self) -> None:
        """
        Initialize the model from another KAN model.

        Raises:
            ValueError: If the model is not initialized.
            RuntimeError: If there is a critical error during initialization.
        """
        try:
            if self.kan_instance is None:
                raise ValueError("Model is not initialized")

            filename = filedialog.askopenfilename(
                filetypes=[("PyTorch Model", "*.pth")]
            )
            if filename:
                self._initialize_from_another_model_file(filename)
        except ValueError as e:
            logging.error("ValueError initializing from another model: %s", e)
            messagebox.showerror(
                "Error", f"ValueError initializing from another model: {e}"
            )
        except RuntimeError as e:
            logging.critical("Critical error initializing from another model: %s", e)
            messagebox.showerror(
                "Error", f"Critical error initializing from another model: %s", e
            )

    def _initialize_from_another_model_file(self, filename: str) -> None:
        """
        Initialize the model from another KAN model.

        Args:
            filename (str): The file path to load the model from.
        """
        other_kan_instance = KANWrapper.load(filename)
        x = torch.normal(0, 1, size=(100, 2))
        self.kan_instance.initialize_from_another_model(other_kan_instance, x)
        messagebox.showinfo(
            "Success", "Model initialized from another model successfully!"
        )


# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    app = KANGUI(root)
    root.mainloop()
