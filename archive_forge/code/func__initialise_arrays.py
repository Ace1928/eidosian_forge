import numpy as np
def _initialise_arrays(self, ignore_n_images, number_of_segments):
    """

        Private function to initialise data storage objects. This includes objects to store the total timesteps
        sampled, the average diffusivity for species in any given segment, and objects to store gradient and intercept from fitting.

        Parameters:
            ignore_n_images (Int): 
                Number of images you want to ignore from the start of the trajectory, e.g. during equilibration
            number_of_segments (Int): 
                Divides the given trajectory in to segments to allow statistical analysis
        
        """
    total_images = len(self.traj) - ignore_n_images
    self.no_of_segments = number_of_segments
    self.len_segments = total_images // self.no_of_segments
    self.timesteps = np.linspace(0, total_images * self.timestep, total_images + 1)
    self.xyz_segment_ensemble_average = np.zeros((self.no_of_segments, self.no_of_types_of_atoms, 3, self.len_segments))
    self.slopes = np.zeros((self.no_of_types_of_atoms, self.no_of_segments, 3))
    self.intercepts = np.zeros((self.no_of_types_of_atoms, self.no_of_segments, 3))
    self.cont_xyz_segment_ensemble_average = 0