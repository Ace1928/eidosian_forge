import numpy as np
def _fit_data(self, x, y):
    """
        Private function that returns slope and intercept for linear fit to mean square diffusion data


        Parameters:
            x (Array of floats): 
                Linear list of timesteps in the calculation
            y (Array of floats): 
                Mean square displacement as a function of time.
        
        """
    slopes = np.zeros(3)
    intercepts = np.zeros(3)
    x_edited = np.vstack([np.array(x), np.ones(len(x))]).T
    for xyz in range(3):
        slopes[xyz], intercepts[xyz] = np.linalg.lstsq(x_edited, y[xyz], rcond=-1)[0]
    return (slopes, intercepts)