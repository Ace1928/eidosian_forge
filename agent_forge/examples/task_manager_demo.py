from agent_forge.agent.task_manager import TaskManager
from agent_forge.core.memory import Memory
from agent_forge.core.model import get_model


def main():
    # Create base components
    model = get_model()
    memory = Memory()

    # Create an agent
    agent = Agent(name="EidosTasker", model=model, memory=memory)

    # Initialize task manager
    task_manager = TaskManager(agent=agent)

    # Create a complex task that breaks down into subtasks
    complex_task = "Research quantum computing applications in cryptography and summarize the findings"

    # Run the task through the manager
    result = task_manager.execute_task(complex_task)

    print("Task execution complete:")
    print(result)


if __name__ == "__main__":
    main()
