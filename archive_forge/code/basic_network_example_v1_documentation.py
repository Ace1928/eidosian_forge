import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import sklearn.model_selection

    Use the learned parameters to predict labels for a dataset X.
    
    Args:
        X (np.ndarray): Input data to predict.
        parameters (Dict[str, np.ndarray]): Parameters of the trained model.
        
    Returns:
        np.ndarray: Predictions (0/1) for the input dataset.
    