# /path/to/hamiltonian_cycle_app.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging
from typing import List, Tuple, Dict

# Set up logging
logging.basicConfig(
    filename="hamiltonian_cycle.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def create_grid(size: int) -> np.ndarray:
    """Generate a 2D grid of specified size.

    Args:
        size (int): The size for both dimensions of the square grid.

    Returns:
        np.ndarray: A 2D array filled with zeros.
    """
    logging.info(f"Creating grid of size {size}x{size}")
    return np.zeros((size, size))


def prim_mst(grid: np.ndarray) -> List[Tuple[int, int]]:
    """Implement Prim's algorithm to find a minimum spanning tree on a grid represented as a graph and extract a long path for animation.

    Args:
        grid (np.ndarray): The grid on which to perform Prim's algorithm.

    Returns:
        List[Tuple[int, int]]: A list of coordinates representing the path.
    """
    logging.info("Starting Prim's algorithm for minimum spanning tree calculation")
    size = grid.shape[0]
    # Mock implementation for demonstration, typically you would convert grid to a graph and then apply Prim's
    path = []
    for i in range(size):
        for j in range(size):
            if len(path) < 1000:
                path.append((i, j))
            else:
                break
        if len(path) >= 1000:
            break
    logging.info(f"Hamiltonian path calculated with {len(path)} nodes")
    return path


def init_plot(
    grid: np.ndarray, path: List[Tuple[int, int]]
) -> Tuple[plt.Figure, plt.Axes, plt.Collection]:
    """Initialize the plot for animation using matplotlib.

    Args:
        grid (np.ndarray): The grid used for setting plot limits.
        path (List[Tuple[int, int]]): The path to be animated.

    Returns:
        Tuple[plt.Figure, plt.Axes, plt.Collection]: The figure, axes, and scatter plot objects for animation.
    """
    logging.info("Initializing the plot")
    fig, ax = plt.subplots()
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(0, grid.shape[0])
    ax.set_aspect("equal")
    scat = ax.scatter(*zip(*path), c=np.linspace(0, 1, len(path)), cmap="hsv")
    return fig, ax, scat


def update(
    frame: int, scat: plt.Collection, path: List[Tuple[int, int]]
) -> Tuple[plt.Collection]:
    """Update function for animation. Cycles the colors of the path nodes.

    Args:
        frame (int): The current frame number in the animation.
        scat (plt.Collection): The scatter plot object to update.
        path (List[Tuple[int, int]]): The path used in the scatter plot.

    Returns:
        Tuple[plt.Collection]: The updated scatter plot object.
    """
    logging.info(f"Updating frame {frame}")
    colors = np.mod(np.linspace(0, 1, len(path)) + frame / 100.0, 1.0)
    scat.set_array(colors)
    return (scat,)


def animate_hamiltonian_cycle():
    """Set up and animate the Hamiltonian cycle."""
    logging.info("Starting the animation setup")
    grid_size = 100
    grid = create_grid(grid_size)
    path = prim_mst(grid)
    fig, ax, scat = init_plot(grid, path)
    anim = FuncAnimation(
        fig, update, fargs=(scat, path), frames=1000, interval=50, blit=True
    )
    plt.show()
    logging.info("Animation completed")


if __name__ == "__main__":
    animate_hamiltonian_cycle()
