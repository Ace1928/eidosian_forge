from agent_forge.agent.agent import Agent
from agent_forge.core.memory import Memory
from agent_forge.core.model import get_model
from agent_forge.core.sandbox import Sandbox


def main():
    # Set up the agent
    model = get_model()
    memory = Memory()
    agent = Agent(name="EidosSandboxTest", model=model, memory=memory)

    # Initialize sandbox
    sandbox = Sandbox()

    # Code to be executed in sandbox
    code = """
    def calculate_fibonacci(n):
        if n <= 1:
            return n
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

    result = calculate_fibonacci(10)
    print(f"The 10th Fibonacci number is: {result}")
    """

    # Execute in sandbox
    result = sandbox.execute(code)

    print("Sandbox execution result:")
    print(result)


if __name__ == "__main__":
    main()
